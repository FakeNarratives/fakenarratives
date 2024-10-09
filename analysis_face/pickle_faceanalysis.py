import os
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append(".")
from project_utils import set_seeds, setup_logging, load_pickle, save_pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Combine face analysis results")
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
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    return parser.parse_args()

def combine_face_analysis(features_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Load all individual analysis results
    face_detection = load_pickle(features_dir / "face_detection_insightface.pkl")
    face_tracking = load_pickle(features_dir / "face_tracking.pkl")
    face_clustering = load_pickle(features_dir / "face_clustering.pkl")
    headgaze = load_pickle(features_dir / "headgaze_3DGazeNet.pkl")
    # headpose = load_pickle(features_dir / "headpose_6DRepNet.pkl")        ## Not needed anymore
    asd_results = load_pickle(features_dir / "asd_light-asd.pkl")
    emotions = load_pickle(features_dir / "face_emotions_deepface.pkl")

    # Create mappings for quick lookups
    face_id_to_track_id = {}
    for track in face_tracking['y']:
        for frame in track['frames']:
            face_id_to_track_id[frame['face_id']] = track['track_id']

    clustering_map = {item['face_id']: item['cluster_id'] for item in face_clustering['y']}
    headgaze_map = {item['face_id']: item['headgaze'] for item in headgaze['y']}
    # headpose_map = {item['face_id']: item['headpose'] for item in headpose['y']}
    asd_map = {track['track_id']: track for track in asd_results['tracks']}
    emotions_map = {item['face_id']: item['emotions'] for item in emotions['y']}

    face_cnt_threshold = sum(1 for face in face_detection['y'] if face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05)

    # print("Face count threshold:", face_cnt_threshold)
    # print("Headgaze count:", len(headgaze_map))
    # print("Emotions count:", len(emotions_map))

    assert face_cnt_threshold == len(headgaze_map), "Face count mismatch with headgaze"
    # assert face_cnt_threshold == len(headpose_map), "Face count mismatch with headpose"
    assert face_cnt_threshold == len(emotions_map), "Face count mismatch with emotions"

    combined_faces = []

    for face in face_detection['y']:
        face_id = face['id']
        track_id = face_id_to_track_id.get(face_id)
        
        asd_info = asd_map.get(track_id, {})
        
        combined_face = {
            "face_id": face_id,
            "time": face['time'],
            "delta_time": face['delta_time'],
            "frame": face['frame'],
            "bbox": face['bbox'],
            "kpss": face['kps'],
            "emb": np.array(face['embedding']),
            "track_id": track_id,
            "cluster_id": clustering_map.get(face_id),
            "gaze": headgaze_map.get(face_id, {}),
            "pose": None,  ## headpose_map.get(face_id, {}),
            "speaking": asd_info.get('is_speaking', False),
            "speaking_ratio": asd_info.get('speaking_ratio', 0),
            "speaking_frames": asd_info.get('speaking_frames', 0),
            "speaking_scores": asd_info.get('smoothed_scores', []),
            "emotions": emotions_map.get(face_id, {})
        }
        
        combined_faces.append(combined_face)

    return combined_faces, face_detection['args']

def process_video(video_path: Path, output_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    combined_faces, detection_args = combine_face_analysis(output_dir)

    output_file = output_dir / "face_analysis.pkl"
    save_pickle({"faces": combined_faces, "args": detection_args}, output_file)

    logging.info(f"Combined face analysis saved to {output_file}")
    logging.info(f"Total faces: {len(combined_faces)}")
    logging.info(f"Speaking faces: {sum(1 for face in combined_faces if face['speaking'])}")

    return combined_faces

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
        output_file = output_dir / "face_analysis.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            combined_faces = process_video(video_path, output_dir, args)

            logging.info(f"Face analysis output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("face_analysis")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args)

    # Log summary
    total = successful + failed
    logging.info(f"Face analysis processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nFace analysis processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()