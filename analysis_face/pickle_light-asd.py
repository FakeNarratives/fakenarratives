import sys
import time
import os
import torch
import argparse
import glob
import subprocess
import warnings
import cv2
import pickle
import numpy as np
import math
import python_speech_features
import logging
from scipy.io import wavfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging

sys.path.append("Light-ASD")
from ASD import ASD

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Light-ASD For Active Speaker Detection")
    parser.add_argument(
        "-v",
        "--videos",
        nargs="+",
        type=str,
        required=True,
        help="Path to input videos"
    )
    parser.add_argument(
        "-o",
        "--pkl_dir",
        type=str,
        required=True,
        help="Path to pkl directory"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="talkset",
        help="talkset or ava"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=10,
        help="Number of workers"
    )
    parser.add_argument(
        "--min_track",
        type=int,
        default=10,
        help="Number of min frames for each shot"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="The start time of the video"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="The duration of the video, when set as 0, will extract the whole video"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the result"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1280,
        help="Max dimension of the video frames"
    )
    return parser.parse_args()

def extract_MFCC(file: str, outPath: str) -> None:
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)
    featuresPath = Path(outPath) / f"{Path(file).stem}.npy"
    np.save(featuresPath, mfcc)

def evaluate_network(files: List[str], s: ASD, args: argparse.Namespace, fps: int) -> List[np.ndarray]:
    allScores = []
    durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
    
    for file in tqdm(files, total=len(files)):
        fileName = Path(file).stem
        _, audio = wavfile.read(Path(args.pycrop_path) / f"{fileName}.wav")
        if len(audio) == 0:
            continue
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        video = cv2.VideoCapture(str(Path(args.pycrop_path) / f"{fileName}.avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = np.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])

        if audioFeature.size == 0 or videoFeature.size == 0:
            logging.warning(f"Skipping file {fileName} due to empty audio or video feature")
            continue
        
        audioFeature = audioFeature[:int(round(length * 100)), :]
        videoFeature = videoFeature[:int(round(length * fps)), :, :]

        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100 : (i + 1) * duration * 100, :]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[i * duration * fps : (i + 1) * duration * fps, :, :]).unsqueeze(0).cuda()

                    if inputA.size(1) == 0 and inputV.size(1) != 0:
                        inputA = torch.zeros((1, inputV.size(1) * 4, inputA.size(2))).cuda()
                    elif inputV.size(1) == 0 and inputA.size(1) != 0:
                        inputV = torch.zeros((1, inputA.size(1) // 4, 112, 112)).cuda()
                    elif inputA.size(1) == 0 and inputV.size(1) == 0:
                        logging.warning(f"Skipping batch {i} due to both inputs being empty")
                        continue

                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    
                    max_length = max(embedA.size(1), embedV.size(1))

                    if embedA.size(1) < max_length:
                        padding = torch.zeros(embedA.size(0), max_length - embedA.size(1), embedA.size(2)).cuda()
                        embedA = torch.cat([embedA, padding], dim=1)

                    if embedV.size(1) < max_length:
                        padding = torch.zeros(embedV.size(0), max_length - embedV.size(1), embedV.size(2)).cuda()
                        embedV = torch.cat([embedV, padding], dim=1)

                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        max_length = max(len(scores) for scores in allScore)
        padded_scores = [scores + [-5] * (max_length - len(scores)) for scores in allScore]
        allScore = np.round((np.mean(np.array(padded_scores), axis=0)), 1).astype(float)
        allScores.append(allScore)

    return allScores

def visualization(tracks: List[Dict[str, Any]], scores: List[np.ndarray], args: argparse.Namespace, vr: List[np.ndarray], fps: int) -> None:
    faces = [[] for _ in range(len(vr))]
    for tidx, track in enumerate(tracks):
        if tidx < len(scores):
            score = scores[tidx]
            for fidx, frame in enumerate(track["track"]["frame"].tolist()):
                s = score[max(fidx - 2, 0):min(fidx + 3, len(score) - 1)]
                s = np.mean(s)
                faces[frame].append({
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                })
    
    firstImage = vr[0]
    fw, fh = firstImage.shape[1], firstImage.shape[0]
    vOut = cv2.VideoWriter(str(Path(args.output_dir) / "video_only.avi"), cv2.VideoWriter_fourcc(*"XVID"), fps, (fw, fh))
    colorDict = {0: 0, 1: 255}
    
    for fidx, image in tqdm(enumerate(vr), total=len(vr)):
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(image,
                          (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                          (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                          (0, clr, 255 - clr), 10)
            cv2.putText(image, f"{txt}", (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, clr, 255 - clr), 5)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vOut.write(image_rgb)
    vOut.release()
    
    command = f"ffmpeg -y -i {Path(args.output_dir) / 'video_only.avi'} -i {Path(args.output_dir) / 'audio.wav'} " \
              f"-threads {args.workers} -c:v copy -c:a copy {Path(args.output_dir) / f'video_out_{args.model}.avi'} -loglevel panic"
    subprocess.call(command, shell=True, stdout=None)

def combine_tracks_and_scores(tracks: List[Dict[str, Any]], scores: List[np.ndarray]) -> List[Dict[str, Any]]:
    assert len(tracks) == len(scores), "Number of tracks and scores must match"
    combined_results = []
    for track, score in zip(tracks, scores):
        smoothed_scores = []
        for i in range(len(score)):
            s = score[max(i-2, 0):min(i+3, len(score))]
            smoothed_scores.append(float(np.mean(s)))
        
        speaking_frames = sum(s >= 0 for s in smoothed_scores)
        speaking_ratio = speaking_frames / len(smoothed_scores)

        combined_track = {
            "track_id": track['track']['track_id'],
            "frames": track['track']['frame'].tolist(),
            "bbox": track['track']['bbox'].tolist(),
            "is_speaking": speaking_ratio > 0.6,
            "speaking_ratio": speaking_ratio,
            "speaking_frames": speaking_frames,
            "mean_score": float(np.mean(smoothed_scores)),
            "original_scores": score,
            "smoothed_scores": smoothed_scores
        }

        combined_results.append(combined_track)
        
    return combined_results

def process_video(video_path: Path, output_dir: Path, args: argparse.Namespace, model: ASD) -> List[Dict[str, Any]]:
    args.output_dir = str(output_dir)
    args.audio_file_path = str(output_dir / 'audio.wav')
    args.pycrop_path = str(output_dir / 'facecrops')

    face_detection_path = output_dir / "face_detection_insightface.pkl"

    if not face_detection_path.exists():
        raise FileNotFoundError(f"Missing face detection file in {face_detection_path}")

    with open(face_detection_path, "rb") as pklfile:
        face_content = pickle.load(pklfile)

    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, face_content["args"]["fps"])
    logging.info(f"Video info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    
    assert frame_width == face_content["args"]["frame_width"]
    assert frame_height == face_content["args"]["frame_height"]

    with open(output_dir / 'tracks.pkl', 'rb') as f:
        vidTracks = pickle.load(f)

    files = glob.glob(f"{args.pycrop_path}/*.avi")
    files.sort()
    scores = evaluate_network(files, model, args, int(fps))

    save_path = output_dir / "scores.pkl"
    with open(save_path, "wb") as fil:
        pickle.dump(scores, fil)
    logging.info(f"Scores extracted and saved in {save_path}")

    combined_results = combine_tracks_and_scores(vidTracks, scores)

    if args.visualize:
        visualization(vidTracks, scores, args, vd, int(fps))

    return combined_results

def process_videos(videos: List[str], pkl_dir: str, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
    successful = 0
    failed = 0
    failed_videos = []

    ## Load model
    model = ASD()
    model.loadParameters(args.pretrain_model)
    logging.info(f"Model {args.pretrain_model} loaded from previous state!")
    model.eval()

    for vi, video in enumerate(videos):
        video_path = Path(video)
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            failed += 1
            failed_videos.append(str(video_path))
            continue
        
        output_dir = Path(pkl_dir) / video_path.stem
        output_file = output_dir / "asd_light-asd.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            combined_results = process_video(video_path, output_dir, args, model)

            with open(output_file, "wb") as f:
                pickle.dump({"tracks": combined_results, "args": vars(args)}, f)

            logging.info(f"ASD results saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("ASD")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    args.pretrain_model = "Light-ASD/weight/finetuning_TalkSet.model" if args.model == "talkset" else "Light-ASD/weight/pretrain_AVA_CVPR.model"

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args)

    # Log summary
    total = successful + failed
    logging.info(f"Active Speaker Detection processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nActive Speaker Detection processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()