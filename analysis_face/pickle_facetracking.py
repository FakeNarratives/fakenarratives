import os
import sys
import torch
import uuid
import time
import pickle
import logging
import argparse
import numpy as np
import cv2
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy import signal

sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Face tracking and crop generation")
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
        "--min_track",
        type=int,
        default=25,
        help="Number of min frames for each shot"
    )
    parser.add_argument(
        "--num_failed_det",
        type=int,
        default=10,
        help="Number of missed detections allowed before tracking is stopped"
    )
    parser.add_argument(
        "--min_face_size",
        type=int,
        default=1,
        help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=0.40,
        help="Scale bounding box"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers")
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1280,
        help="Max dimension of the video frames"
    )
    return parser.parse_args()

def tracking_pkl(tracks: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]
    return {
        "y": tracks,
        "args": args_dict,
    }

def bb_intersection_over_union(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def normalize_to_pixel(bbox: Dict[str, float], frame_width: int, frame_height: int) -> List[float]:
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    x1 = x * frame_width
    y1 = y * frame_height
    x2 = (x + w) * frame_width
    y2 = (y + h) * frame_height
    return [x1, y1, x2, y2]

def track_shot(args: argparse.Namespace, shotFaces: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        enhanced_track = {
            'frame': [],
            'bbox': [],
            'frames': []
        }
        for frameFaces in shotFaces:
            for face in frameFaces:
                if not track:
                    track.append(face)
                    frameFaces.remove(face)
                    enhanced_track['frame'].append(face['frame'])
                    enhanced_track['bbox'].append(face['bbox'])
                    enhanced_track['frames'].append({
                        'frame_number': face['frame'],
                        'face_id': face['id'],
                        'embedding': face['embedding']
                    })
                elif face['frame'] - track[-1]['frame'] <= args.num_failed_det:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        enhanced_track['frame'].append(face['frame'])
                        enhanced_track['bbox'].append(face['bbox'])
                        enhanced_track['frames'].append({
                            'frame_number': face['frame'],
                            'face_id': face['id'],
                            'embedding': face['embedding']
                        })
                        continue
                else:
                    break
        if not track:
            break
        elif len(track) > args.min_track:
            frameNum = np.array(enhanced_track['frame'])
            bboxes = np.array(enhanced_track['bbox'])
            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:, 2] - bboxesI[:, 0]), np.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.min_face_size:
                track_id = str(uuid.uuid4())
                enhanced_track['track_id'] = track_id
                enhanced_track['frame'] = frameI
                enhanced_track['bbox'] = bboxesI
                
                tracks.append(enhanced_track)
    return tracks

def crop_video(args: argparse.Namespace, track: Dict[str, Any], vr: List[np.ndarray], fps: float, cropFile: str, audioFilePath: str) -> Dict[str, Any]:
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (224, 224))
    
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    
    dets['s'] = np.array(signal.medfilt(dets['s'], kernel_size=13))
    dets['x'] = np.array(signal.medfilt(dets['x'], kernel_size=13))
    dets['y'] = np.array(signal.medfilt(dets['y'], kernel_size=13))
    
    cs = args.crop_scale
    frame_nums = np.array(track['frame'])
    
    for fidx, frame_num in enumerate(frame_nums):
        image = cv2.cvtColor(vr[frame_num], cv2.COLOR_RGB2BGR)
        
        bs = dets['s'][fidx]
        my = dets['y'][fidx]
        mx = dets['x'][fidx]
        
        y1 = int(my - bs)
        y2 = int(my + bs * (1 + 2 * cs))
        x1 = int(mx - bs * (1 + cs))
        x2 = int(mx + bs * (1 + cs))
        
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - image.shape[0])
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - image.shape[1])
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[110, 110, 110])
        
        crop_y1 = max(0, y1 + pad_top)
        crop_y2 = min(image.shape[0], y2 + pad_top)
        crop_x1 = max(0, x1 + pad_left)
        crop_x2 = min(image.shape[1], x2 + pad_left)
        
        face = image[crop_y1:crop_y2, crop_x1:crop_x2]
        face_resized = cv2.resize(face, (224, 224))
        
        vOut.write(face_resized)
    
    vOut.release()
    
    subaudioFilePath = cropFile + '.wav'
    audioStart = frame_nums[0] / fps
    audioEnd = (frame_nums[-1] + 1) / fps
    
    command = (f"ffmpeg -y -i {audioFilePath} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads {args.workers} "
               f"-ss {audioStart:.3f} -to {audioEnd:.3f} {subaudioFilePath} -loglevel panic")
    subprocess.call(command, shell=True)

    command = (f"ffmpeg -y -i {cropFile}t.avi -i {subaudioFilePath} -threads {args.workers} "
               f"-c:v copy -c:a copy {cropFile}.avi -loglevel panic")
    subprocess.call(command, shell=True)
    
    os.remove(cropFile + 't.avi')
    
    return {'track': track, 'proc_track': dets, 'video_file': cropFile + '.avi', 'audio_file': subaudioFilePath}

def process_video(video_path: Path, output_dir: Path, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pycropPath = output_dir / 'facecrops'
    pycropPath.mkdir(parents=True, exist_ok=True)

    audioFilePath = output_dir / 'audio.wav'
    if not audioFilePath.exists():
        raise FileNotFoundError(f"Audio file {audioFilePath} does not exist.")

    face_detection_path = output_dir / "face_detection_insightface.pkl"
    shot_detection_path = output_dir / "transnet_shotdetection.pkl"

    if not face_detection_path.exists() or not shot_detection_path.exists():
        raise FileNotFoundError(f"Missing shot or face detection file in {output_dir}")

    with open(face_detection_path, "rb") as pklfile:
        face_content = pickle.load(pklfile)
    with open(shot_detection_path, "rb") as pklfile:
        shot_content = pickle.load(pklfile)

    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, face_content["args"]["fps"])
    logging.info(f"Video info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    
    assert frame_width == face_content["args"]["frame_width"]
    assert frame_height == face_content["args"]["frame_height"]

    faces_by_frame = defaultdict(list)
    for face in face_content["y"]:
        if face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05:
            frame_index = face["frame"]
            pixel_bbox = normalize_to_pixel(face['bbox'], frame_width, frame_height)
            faces_by_frame[frame_index].append({
                'id': face['id'],
                'frame': frame_index,
                'bbox': pixel_bbox,
                'conf': face['det_score'],
                'embedding': face['embedding']
            })

    max_frame = max(faces_by_frame.keys())
    faces_list = [[] for _ in range(max_frame + 1)]
    for frame_index, face_list in faces_by_frame.items():
        faces_list[frame_index] = face_list

    allTracks = []
    for shot in shot_content["output_data"]["shots"]:
        if shot["end_frame"] - shot["start_frame"] >= args.min_track:
            allTracks.extend(track_shot(args, faces_list[shot["start_frame"]:shot["end_frame"]]))
    logging.info(f"Detected {len(allTracks)} tracks")

    vidTracks = []
    for ii, track in tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(args, track, vd, fps, str(pycropPath / f'{ii:05d}'), str(audioFilePath)))

    return allTracks, vidTracks

def process_videos(videos: List[str], pkl_dir: str, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
    successful = 0
    failed = 0
    failed_videos = []

    for vi, video in enumerate(videos):
        video_path = Path(video)
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            failed += 1
            failed_videos.append(str(video_path))
            continue
        
        output_dir = Path(pkl_dir) / video_path.stem
        output_file = output_dir / "face_tracking.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            allTracks, vidTracks = process_video(video_path, output_dir, args)

            tracks_path = output_dir / 'tracks.pkl'
            with open(tracks_path, 'wb') as fil:
                pickle.dump(vidTracks, fil)
            logging.info(f"Face Crop and saved in {output_dir / 'facecrops'} tracks")

            face_tracking_data = tracking_pkl(allTracks, args)
            with open(output_file, 'wb') as f:
                pickle.dump(face_tracking_data, f)

            logging.info(f"Face tracking output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1

            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("face_tracking")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args)

    # Log summary
    total = successful + failed
    logging.info(f"Face tracking processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nFace tracking processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()