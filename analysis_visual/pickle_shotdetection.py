import os
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transnetv2 import TransNetV2

sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging, save_pickle, load_pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Performs shot detection in a video using TransNetV2.")
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
        "--max_dimension",
        type=int,
        default=48,
        help="Max dimension of the video frames"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of workers"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    return parser.parse_args()

def get_model() -> TransNetV2:
    model = TransNetV2("TransNetV2/inference/transnetv2-weights/")
    logging.info("TransNetV2 Model Loaded")
    return model

def process_video(video_path: Path, model: TransNetV2, args: argparse.Namespace) -> Dict[str, Any]:
    shot_dict = {
        "github_repo": "https://github.com/soCzech/TransNetV2",
        "commit_id": "85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed", 
        "parameters": "default",
        "video_file": str(video_path),
        "output_data": {"shots": []}
    }

    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, args.fps)
    logging.info(f"Video info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    all_frames = np.array([vd[i] for i in range(len(vd))])
    logging.info(f"All Frames: {all_frames.shape}")

    single_frame_predictions, _ = model.predict_frames(all_frames)

    shots = model.predictions_to_scenes(single_frame_predictions)

    for shot in shots:
        start_frame, end_frame = shot
        start_time = start_frame / args.fps
        end_time = end_frame / args.fps

        shot_dict["output_data"]["shots"].append({
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start": start_time,
            "end": end_time
        })

    return shot_dict

def process_videos(videos: List[str], pkl_dir: str, model: TransNetV2, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
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
        output_file = output_dir / "transnet_shotdetection.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            ## Read and print number of shots
            shot_dict = load_pickle(output_file)
            logging.info(f"Number of shots in video: {len(shot_dict['output_data']['shots'])}")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            shot_dict = process_video(video_path, model, args)
            save_pickle(shot_dict, output_file)

            logging.info(f"Shot detection output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("shot_detection")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    model = get_model()

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, model, args)

    # Log summary
    total = successful + failed
    logging.info(f"Shot detection processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nShot detection processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()

# Before running the script, export the following environment variable in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nfs/home/cheemag/miniconda3/envs/faken_v/lib/:/nfs/home/cheemag/miniconda3/envs/faken_v/lib/python3.10/site-packages/torch/lib/