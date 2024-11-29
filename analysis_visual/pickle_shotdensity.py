import os
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse
import logging
import sys
import math
from sklearn.neighbors import KernelDensity
from typing import List, Dict, Any, Tuple
sys.path.append(".")
from project_utils import set_seeds, setup_logging, save_pickle, load_pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Performs shot detection in a video using TransNetV2.")
    parser.add_argument(
        "-v", "--videos", nargs="+", type=str, required=True, help="path to the video files"
    )
    parser.add_argument(
        "-o", "--pkl_dir", type=str, required=True, help="path to the output folder"
    )
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=25.0,
        help="Frames per second"
    )
    parser.add_argument(
        "-b", "--bandwidth",
        type=int,
        default=10,
        help="Bandwidth for KDE"
    )
    args = parser.parse_args()
    return args


def process_video(video_path, shots_data, fps, bandwidth):
    """
    Use Kernel Density Estimation to compute shot density

    Args:
        video_path (str): path to video
        output_path (str): path to save output

    Returns:
        {"y": shot_density, "time": time_stamps, "delta_time": 1/fps}
    """

    last_shot_end = 0
    shots = []
    for shot in shots_data:
        shots.append(shot["start"])

        if shot["end"]> last_shot_end:
            last_shot_end = shot["end"]

    time = np.linspace(0, last_shot_end, math.ceil(last_shot_end * fps) + 1)[:, np.newaxis]
    shots = np.asarray(shots).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(shots)
    log_dens = kde.score_samples(time)
    shot_density = np.exp(log_dens)
    shot_density = (shot_density - shot_density.min()) / (shot_density.max() - shot_density.min())

    output_data = {
        "video_file": video_path,
        "parameters": {"fps": fps, "bandwidth": bandwidth},
        "output_data": {
                    "y": shot_density.squeeze(),
                    "time": time.squeeze().astype(np.float64),
                    "delta_time": 1 / fps
                }
    }

    return output_data


def process_videos(videos: List[str], pkl_dir: str, args: Any) -> Tuple[int, int, List[str]]:
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
        shot_file = output_dir / "transnet_shotdetection.pkl"

        if not shot_file.exists():
            logging.warning(f"Shot boundary detection file not found: {shot_file}")
            failed += 1
            failed_videos.append(str(video_path))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "shot_density.pkl"

        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")

            with open(shot_file, 'rb') as pkl:
                shots_data = pickle.load(pkl)["output_data"]["shots"]

            shot_density = process_video(video_path, shots_data, args.fps, args.bandwidth)
            save_pickle(shot_density, output_file)

            logging.info(f"Shot density output saved to: [{vi+1}/{len(videos)}]: {output_file}")

            successful += 1
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos


def main():
    args = parse_args()
    log_file = setup_logging("shot_density")

    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args)

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