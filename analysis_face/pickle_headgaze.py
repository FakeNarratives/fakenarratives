import os
import sys
import torch
import uuid
import time
import pickle
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging

sys.path.append("3DGazeNet/")
from models import GazeModel
from handler import GazeHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Headgaze estimation using 3DGazeNet")
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
        "--ckpt_path",
        type=str,
        default="3DGazeNet/assets/latest_a.ckpt",
        help="Path to checkpoint file"
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
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size for processing")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers")
    return parser.parse_args()

def headgaze_pkl(headgazes: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]
    return {
        "y": headgazes,
        "args": args_dict,
    }

def process_video(video_path: Path, output_dir: Path, handler: GazeHandler, args: argparse.Namespace) -> List[Dict[str, Any]]:
    face_detection_path = output_dir / "face_detection_insightface.pkl"
    
    if not face_detection_path.exists():
        raise FileNotFoundError(f"Missing file: {face_detection_path}")

    with open(face_detection_path, "rb") as pklfile:
        face_content = pickle.load(pklfile)

    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, face_content["args"]["fps"])
    logging.info(f"Video info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    
    assert frame_width == face_content["args"]["frame_width"]
    assert frame_height == face_content["args"]["frame_height"]

    headgazes = []
    batch_frames = []
    batch_faces = []
    batch_frame_indices = []
    valid_face_cnt = 0

    def process_batch():
        nonlocal valid_face_cnt, headgazes
        if not batch_frames:
            return

        results = handler.get_batch(batch_faces, batch_frames)

        assert len(results) == len(batch_faces) == len(batch_frame_indices), "Batch size mismatch"

        for frame_results, frame_faces, frame_index in zip(results, batch_faces, batch_frame_indices):
            eye_kps_batch = [result[2] for result in frame_results]
            gaze_preds = handler.get_gaze_batch(eye_kps_batch)

            assert len(frame_faces) == len(gaze_preds), "Face count mismatch with gaze predictions"

            for face, (gaze_pred_l_rad, gaze_pred_r_rad) in zip(frame_faces, gaze_preds):
                gaze_pred_l_deg, gaze_pred_r_deg = np.degrees(gaze_pred_l_rad), np.degrees(gaze_pred_r_rad)

                headgazes.append({
                    "face_id": face["id"],
                    "frame": frame_index,
                    "time": face["time"],
                    "headgaze": {
                        "left_gaze_rad": gaze_pred_l_rad.tolist(),
                        "right_gaze_rad": gaze_pred_r_rad.tolist(),
                        "left_gaze_deg": gaze_pred_l_deg.tolist(),
                        "right_gaze_deg": gaze_pred_r_deg.tolist(),
                    },
                })

    for frame_index, image in enumerate(tqdm(vd, desc="Processing frames")):
        frame_faces = [
            face for face in face_content["y"]
            if face["frame"] == frame_index and face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05
        ]

        if not frame_faces:
            continue

        valid_face_cnt += len(frame_faces)

        # Rescale normalized bounding boxes and keypoints to the original image size
        for face in frame_faces:
            face["bbox"]["w"] *= image.shape[1]
            face["bbox"]["h"] *= image.shape[0]
            face["bbox"]["x"] *= image.shape[1]
            face["bbox"]["y"] *= image.shape[0]
            face["kps"] = np.array(face["kps"]) * [image.shape[1], image.shape[0]]

        batch_frames.append(image)
        batch_faces.append(frame_faces)
        batch_frame_indices.append(frame_index)

        if len(batch_frames) == args.batch_size:
            process_batch()
            batch_frames = []
            batch_faces = []
            batch_frame_indices = []

    # Process any remaining frames in the last batch
    if batch_frames:
        process_batch()

    logging.info(f"Detected {valid_face_cnt} valid faces and processed {len(headgazes)} headgazes in video")
    assert valid_face_cnt == len(headgazes), f"Face count mismatch with headgaze: {valid_face_cnt} != {len(headgazes)}"

    return headgazes

def process_videos(videos: List[str], pkl_dir: str, handler: GazeHandler, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
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
        output_file = output_dir / "headgaze_3DGazeNet.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            headgazes = process_video(video_path, output_dir, handler, args)

            with open(output_file, "wb") as f:
                output_dict = headgaze_pkl(headgazes, args)
                pickle.dump(output_dict, f)

            logging.info(f"Headgaze output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("headgaze")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    handler = GazeHandler(args.ckpt_path)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, handler, args)

    # Log summary
    total = successful + failed
    logging.info(f"Headgaze estimation processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nHeadgaze estimation processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()