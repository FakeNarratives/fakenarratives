import os
from pathlib import Path
import argparse
import subprocess
import logging
from datetime import datetime
from typing import List, Tuple
import sys
sys.path.append(".")
from project_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio wav files from videos using ffmpeg")
    parser.add_argument(
        "-v",
        "--videos",
        type=str,
        required=True,
        nargs="+",
        help="Path to input videos",
    )
    parser.add_argument(
        "-o",
        "--pkl_dir", type=str, required=True, help="Path to pkl directory"
    )
    parser.add_argument("-r", "--rewrite", action="store_true", help="Rewrite existing files")
    return parser.parse_args()

def extract_audio(video_path: Path, output_path: Path, rewrite: bool) -> bool:
    if output_path.exists() and not rewrite:
        logging.info(f"Audio file already exists for {video_path}. Skipping.")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        "ffmpeg", "-y", "-i", str(video_path), "-qscale:a", "0", 
        "-ac", "1", "-vn", "-threads", "4", "-ar", "16000", 
        str(output_path), "-loglevel", "panic"
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Extracted audio from {video_path} to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract audio from {video_path}: {e}")
        logging.error(f"ffmpeg stderr: {e.stderr}")
        return False

def process_videos(videos: List[str], pkl_dir: str, rewrite: bool) -> Tuple[int, int, List[str]]:
    successful = 0
    failed = 0
    failed_videos = []

    for video in videos:
        video_path = Path(video)
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            failed += 1
            failed_videos.append(str(video_path))
            continue
        
        output_path = Path(pkl_dir) / video_path.stem / "audio.wav"
        if extract_audio(video_path, output_path, rewrite):
            successful += 1
        else:
            failed += 1
            failed_videos.append(str(video_path))

    return successful, failed, failed_videos

def main():
    script_path = Path(__file__).resolve()
    log_file = setup_logging("audio_extraction")
    
    args = parse_args()
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args.rewrite)

    # Log summary
    total = successful + failed
    logging.info(f"Audio extraction complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed extractions:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nAudio extraction summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful extractions: {successful}")
    print(f"Failed extractions: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed extractions.")

if __name__ == "__main__":
    main()