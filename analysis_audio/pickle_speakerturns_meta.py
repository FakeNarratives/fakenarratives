import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import sys
sys.path.append(".")
from project_utils import set_seeds, setup_logging, load_pickle, save_pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Combine speaker turns with ASD results, speaker roles, and news situations")
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
        "-t",
        "--threshold",
        type=float,
        default=0.6,
        help="Threshold for determining active speaker turn based on the ratio of speaking duration to turn duration"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    return parser.parse_args()

def sort_tracks_by_start_time(asd_tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort ASD tracks by their start time."""
    return sorted(asd_tracks, key=lambda x: x['frames'][0])

def find_overlapping_tracks(turn: Dict[str, Any], sorted_asd_tracks: List[Dict[str, Any]], fps: float) -> List[Tuple[Dict[str, Any], float]]:
    """
    Find all face tracks that overlap with a given speaker turn.
    
    Args:
    turn: A dictionary containing 'start' and 'end' times of the speaker turn.
    sorted_asd_tracks: List of face tracks sorted by start time.
    fps: Frames per second of the video.
    
    Returns:
    A list of tuples, each containing an overlapping track and its overlap duration,
    sorted by overlap duration in descending order.
    """
    turn_start, turn_end = turn['start'], turn['end']
    overlapping_tracks = []
    for track in sorted_asd_tracks:
        track_start = track['frames'][0] / fps
        track_end = track['frames'][-1] / fps
        if track_start >= turn_end:
            break  # No need to check further tracks
        if track_end > turn_start and track_start < turn_end:
            overlap_start = max(turn_start, track_start)
            overlap_end = min(turn_end, track_end)
            overlap_duration = overlap_end - overlap_start
            overlapping_tracks.append((track, overlap_duration))
    
    return sorted(overlapping_tracks, key=lambda x: x[1], reverse=True)

def calculate_speaking_duration(turn: Dict[str, Any], track: Dict[str, Any], fps: float) -> float:
    """
    Calculate the duration of speaking frames within the overlap of a turn and a face track.
    
    Args:
    turn: A dictionary containing 'start' and 'end' times of the speaker turn.
    track: A dictionary containing information about a face track.
    fps: Frames per second of the video.
    
    Returns:
    The duration (in seconds) of speaking frames within the overlap.
    """
    turn_start, turn_end = turn['start'], turn['end']
    track_start = track['frames'][0] / fps
    track_end = track['frames'][-1] / fps
    overlap_start = max(turn_start, track_start)
    overlap_end = min(turn_end, track_end)
    
    overlap_frames = int((overlap_end - overlap_start) * fps)
    speaking_frames = int(overlap_frames * track['speaking_ratio'])
    
    return speaking_frames / fps

def process_video(video_path: Path, output_dir: Path, threshold: float) -> bool:
    try:
        # Load face detection
        face_detection = load_pickle(output_dir / "face_detection_insightface.pkl")
        fps = face_detection['args']['fps']

        # Load ASR results
        asr_path = output_dir / "asr_whisperx.pkl"
        asr_data = load_pickle(asr_path)
        speaker_turns = asr_data['output_data']['speaker_turns']

        logging.info(f"Loaded {len(speaker_turns)} speaker turns")

        # Load ASD results
        asd_path = output_dir / "asd_light-asd.pkl"
        asd_data = load_pickle(asd_path)

        # Sort ASD tracks by start time
        sorted_asd_tracks = sort_tracks_by_start_time(asd_data['tracks'])

        # Load speaker roles and news situations
        speaker_roles = load_pickle(output_dir / "icmr_speaker_roles.pkl")
        news_situations = load_pickle(output_dir / "icmr_situations.pkl")

        assert len(speaker_turns) == len(speaker_roles) == len(news_situations), "Mismatch in the number of turns, roles, and situations"

        # Update speaker turns
        updated_speaker_turns = []
        for i, turn in enumerate(tqdm(speaker_turns, desc=f"Processing {video_path.stem}")):
            updated_turn = {
                'start': turn['start'],
                'end': turn['end'],
                'speaker': turn['speaker']
            }

            overlapping_tracks = find_overlapping_tracks(turn, sorted_asd_tracks, fps)
            
            if overlapping_tracks:
                largest_track, _ = overlapping_tracks[0]
                speaking_duration = calculate_speaking_duration(turn, largest_track, fps)
                turn_duration = turn['end'] - turn['start']
                active_ratio = speaking_duration / turn_duration

                updated_turn['active'] = active_ratio > threshold
                updated_turn['active_ratio'] = active_ratio
                updated_turn['active_track_id'] = largest_track['track_id']
            else:
                updated_turn['active'] = False
                updated_turn['active_ratio'] = 0
                updated_turn['active_track_id'] = None

            updated_turn['role_l0'] = speaker_roles[i]['role_l0']
            updated_turn['role_l1'] = speaker_roles[i]['role_l1']
            updated_turn['situation'] = news_situations[i]['situation']

            updated_speaker_turns.append(updated_turn)

        # Create the new video_feat_dict
        video_feat_dict = {
            "github_repo": "None",
            "commit_id": "None",
            "parameters": "default",
            "video_file": str(video_path),
            "output_data": updated_speaker_turns
        }

        # Save the new pickle file
        new_pickle_path = output_dir / "speaker_turns_meta.pkl"
        save_pickle(video_feat_dict, new_pickle_path)
        logging.info(f"Updated speaker turns and saved to {new_pickle_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        return False

def process_videos(videos: List[str], pkl_dir: str, threshold: float, rewrite: bool) -> Tuple[int, int, List[str]]:
    successful = 0
    failed = 0
    failed_videos = []

    for vi, video in enumerate(videos):
        video_path = Path(video)
        if not video_path.exists():
            failed += 1
            failed_videos.append(str(video_path))
            continue
        
        output_dir = Path(pkl_dir) / video_path.stem
        output_file = output_dir / "speaker_turns_meta.pkl"
        
        if output_file.exists() and not rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
        if process_video(video_path, output_dir, threshold):
            successful += 1
        else:
            failed += 1
            failed_videos.append(str(video_path))

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("speaker_turns_meta")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args.threshold, args.rewrite)

    # Log summary
    total = successful + failed
    logging.info(f"Speaker turns meta processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nSpeaker turns meta processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()