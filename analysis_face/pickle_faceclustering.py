import os
import sys
import torch
import time
import pickle
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Face clustering using DBSCAN")
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
        "--metric",
        type=str,
        default="cosine",
        help="Distance metric for clustering"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="DBSCAN epsilon (less epsilon, more clusters)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize clusters"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of workers"
    )
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1280,
        help="Max dimension of the video frames"
    )
    return parser.parse_args()

def process_tracks(face_tracking: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Process tracks to get embeddings"""
    track_embeddings = {}
    for track in face_tracking['y']:
        if track['frames']:
            embeddings = [frame['embedding'] for frame in track['frames'] if frame['embedding'] is not None]
            if embeddings:
                track_embeddings[track['track_id']] = np.mean(embeddings, axis=0)
    
    logging.info(f"Number of tracks processed: {len(track_embeddings)}")
    return track_embeddings

def improved_clustering(embeddings: np.ndarray, eps: float = 0.5, metric: str = 'cosine') -> np.ndarray:
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Apply PCA
    pca = PCA(n_components=0.95)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    logging.info(f"PCA: {embeddings_pca.shape}")

    # Use DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=2, metric=metric)
    clusters = dbscan.fit_predict(embeddings_pca)

    return clusters

def extract_face_image(bbox: List[float], frame: np.ndarray) -> np.ndarray:
    """Extract and resize face image from frame"""
    h, w, _ = frame.shape
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Ensure the coordinates are within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    face_img = frame[y1:y2, x1:x2]
    return cv2.resize(face_img, (224, 224))

def visualize_clusters(face_tracking: Dict[str, Any], clusters: np.ndarray, unique_clusters: np.ndarray, output_dir: Path, vr: List[np.ndarray]) -> None:
    """Visualize clusters by creating grids of face images"""
    cluster_faces = defaultdict(list)
    for track, cluster in zip(face_tracking['y'], clusters):
        if track['frames']:
            mid_frame = track['frames'][len(track['frames'])//2]
            frame_image = vr[mid_frame['frame_number']]
            face_img = extract_face_image(mid_frame['bbox'], frame_image)
            cluster_faces[cluster].append(face_img)

    for cluster_id in unique_clusters:
        faces = cluster_faces[cluster_id]
        num_faces = min(len(faces), 25)  # Limit to 25 faces per cluster
        grid_size = int(np.ceil(np.sqrt(num_faces)))
        grid = np.zeros((grid_size * 224, grid_size * 224, 3), dtype=np.uint8)
        
        for i, face in enumerate(faces[:num_faces]):
            row = i // grid_size
            col = i % grid_size
            grid[row*224:(row+1)*224, col*224:(col+1)*224] = face

        output_path = output_dir / f"cluster_{cluster_id}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        logging.info(f"Cluster {cluster_id} visualization saved to: {output_path}")

def process_video(video_path: Path, output_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    face_tracking_path = output_dir / "face_tracking.pkl"
    face_detection_path = output_dir / "face_detection_insightface.pkl"

    if not face_detection_path.exists() or not face_tracking_path.exists():
        raise FileNotFoundError(f"Missing face tracking or detection file in {output_dir}")

    with open(face_detection_path, "rb") as pklfile:
        face_content = pickle.load(pklfile)

    with open(face_tracking_path, "rb") as pklfile:
        face_tracking = pickle.load(pklfile)

    # Process tracks to get embeddings
    track_embeddings = process_tracks(face_tracking)

    # Prepare data for clustering
    embeddings = list(track_embeddings.values())

    embeddings = np.array(normalize(embeddings))
    logging.debug(f"Perform clustering ... shape {embeddings.shape}")

    # Perform clustering
    clusters = improved_clustering(embeddings, eps=args.eps, metric=args.metric)
    unique_clusters = np.unique(clusters)
    num_clusters = len(unique_clusters)
    logging.info(f"Number of clusters: {num_clusters}")

    # Visualize clusters if requested
    if args.visualize:
        vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, face_content["args"]["fps"])
        logging.info(f"Video info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
        assert frame_width == face_content["args"]["frame_width"]
        assert frame_height == face_content["args"]["frame_height"]

        vis_output_dir = output_dir / "cluster_visualizations"
        vis_output_dir.mkdir(parents=True, exist_ok=True)
        visualize_clusters(face_tracking, clusters, unique_clusters, vis_output_dir, vr)

    # Prepare clustering results in the desired format
    clustering_results = []
    for track, cluster in zip(face_tracking['y'], clusters):
        for frame in track['frames']:
            clustering_results.append({
                "face_id": frame['face_id'],
                "track_id": track['track_id'],
                "cluster_id": int(cluster)
            })

    return clustering_results

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
        output_file = output_dir / "face_clustering.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            clustering_results = process_video(video_path, output_dir, args)

            with open(output_file, "wb") as f:
                output_dict = {"y": clustering_results, "args": vars(args)}
                pickle.dump(output_dict, f)

            logging.info(f"Face clustering output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("face_clustering")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args)

    # Log summary
    total = successful + failed
    logging.info(f"Face clustering processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nFace clustering processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()