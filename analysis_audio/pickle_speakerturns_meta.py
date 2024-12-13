import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import timedelta
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
        default=0.5,
        help="Threshold for determining active speaker turn based on the ratio of speaking duration to turn duration"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    return parser.parse_args()

def find_overlapping_faces(reference_turn: Dict[str, Any], faces: List[Dict[str, Any]], fps: float):
    ref_start, ref_end = reference_turn["start"], reference_turn["end"]

    overlapping_faces = []
    for face in faces:
        face_time = face["time"] ## Detected at this time

        if face_time < ref_start or face_time > ref_end:
            continue
        
        if face["speaking"]:
            overlapping_faces.append({"face_id": face["face_id"], "time": face_time, "cluster_id": face["cluster_id"], 
                                    "speaking_ratio": face["speaking_ratio"], "speaking": face["speaking"], "bbox": face["bbox"]})

    unique_faces = {}
    for face in overlapping_faces:
        if face["cluster_id"] not in unique_faces and face["cluster_id"] is not None:
            unique_faces[face["cluster_id"]] = [face]
        else:
            unique_faces[face["cluster_id"]].append(face)

    largest_duration = 0
    largest_cluster_id = None
    for cluster_id, faces in unique_faces.items():
        time_diff = faces[-1]["time"] - faces[0]["time"]
        # fc_idx = int(len(faces)/2)
        # print(cluster_id, faces[fc_idx]["time"], faces[fc_idx]["speaking_ratio"], faces[fc_idx]["speaking"])
        if time_diff > largest_duration:
            largest_duration = time_diff
            largest_cluster_id = cluster_id

    if largest_cluster_id is None:
        return None, None, None
    
    return largest_cluster_id, largest_duration, unique_faces[face["cluster_id"]]
    

def process_video(video_path: Path, output_dir: Path, threshold: float) -> bool:
    try:
        # Load ASR results
        asr_path = output_dir / "asr_whisperx.pkl"
        asr_data = load_pickle(asr_path)
        speaker_turns = asr_data['output_data']['speaker_turns']

        logging.info(f"Loaded {len(speaker_turns)} speaker turns")

        # Load combined face analysis results for cluster based active speaker turns
        face_path = output_dir / "face_analysis.pkl" 
        face_data = load_pickle(face_path)
        faces, fps = face_data["faces"], face_data["args"]["fps"]

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

            face_cluster_id, largest_face_duration, largest_face = find_overlapping_faces(turn, faces, fps)
            
            if face_cluster_id is not None:
                turn_duration = turn['end'] - turn['start']
                active_ratio = largest_face_duration / turn_duration
                updated_turn['active'] = active_ratio > threshold
                updated_turn['active_ratio'] = active_ratio
                updated_turn['face'] = {"first_appearance": largest_face[0], "last_appearance": largest_face[-1]}
            else:
                updated_turn['active'] = False
                updated_turn['active_ratio'] = 0
                updated_turn['face'] = None

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