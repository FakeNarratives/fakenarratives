import argparse
import logging
import numpy as np
import os
import cv2
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from collections import defaultdict
from video_utils import read_video_and_get_info
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Face clustering using Agglomerative Clustering")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("--metric", type=str, default="cosine", help="distance metric")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon | less epsilon, more clusters")
    parser.add_argument("--visualize", action="store_true", help="visualize clusters")
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )
    return parser.parse_args()



def process_tracks(face_tracking):
    """Process tracks to get embeddings"""
    track_embeddings = {}
    for track in face_tracking['tracks']:
        if track['frames']:
            embeddings = [frame['embedding'] for frame in track['frames'] if frame['embedding'] is not None]
            if embeddings:
                track_embeddings[track['track_id']] = np.mean(embeddings, axis=0)
    
    logger.info(f"Number of tracks processed: {len(track_embeddings)}")
    return track_embeddings


def improved_clustering(embeddings, eps=0.5, metric='cosine'):
    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Apply PCA
    pca = PCA(n_components=0.95)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    print(f"PCA: {embeddings_pca.shape}")

    # Use DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=2, metric=metric)
    clusters = dbscan.fit_predict(embeddings_pca)

    return clusters


def visualize_clusters(face_tracking, clusters, unique_clusters, output_dir, vr):
    """Visualize clusters by creating grids of face images"""

    cluster_faces = defaultdict(list)
    for track, cluster in zip(face_tracking['tracks'], clusters):
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

        output_path = os.path.join(output_dir, f"cluster_{cluster_id}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        logger.info(f"Cluster {cluster_id} visualization saved to: {output_path}")


def extract_face_image(bbox, frame):
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


def main():
    args = parse_args()
    
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        face_tracking_path = os.path.join(output_dir, "face_tracking.pkl")
        face_detection_path = os.path.join(output_dir, "face_detection_insightface.pkl")

        if not os.path.isfile(face_detection_path) or not os.path.isfile(face_tracking_path):
            logging.error(f"\tMissing face tracking or detection file in {output_dir}")
            continue

        with open(face_detection_path, "rb") as pklfile:
            face_content = pickle.load(pklfile)

        with open(face_tracking_path, "rb") as pklfile:
            face_tracking = pickle.load(pklfile)

        # Process tracks to get embeddings
        track_embeddings = process_tracks(face_tracking)

        # Prepare data for clustering
        embeddings = list(track_embeddings.values())

        embeddings = np.array(normalize(embeddings))
        logger.debug(f"\tPerform clustering ... shape {embeddings.shape}")

        # Perform clustering
        clusters = improved_clustering(embeddings, eps=args.eps, metric=args.metric)
        unique_clusters = np.unique(clusters)
        num_clusters = len(unique_clusters)
        logger.info(f"\tNumber of clusters: {num_clusters}")

        # Visualize clusters if requested
        if args.visualize:
            # get frames from video
            vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, face_content["args"]["fps"])
            logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
            assert frame_width == face_content["args"]["frame_width"]
            assert frame_height == face_content["args"]["frame_height"]

            vis_output_dir = os.path.join(output_dir, "cluster_visualizations")
            os.makedirs(vis_output_dir, exist_ok=True)
            visualize_clusters(face_tracking, clusters, unique_clusters, vis_output_dir, vr)

        # Prepare clustering results in the desired format
        clustering_results = []
        for track, cluster in zip(face_tracking['tracks'], clusters):
            for frame in track['frames']:
                clustering_results.append({
                    "face_id": frame['face_id'],
                    "track_id": track['track_id'],
                    "cluster_id": int(cluster)
                })

        # Save clustering results
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        output_path = os.path.join(output_dir, "face_clustering.pkl")
        with open(output_path, 'wb') as f:
            output_dict = {"y": clustering_results, "args": vars(args)}
            pickle.dump(output_dict, f)
        
        logger.info(f"\tOutput written to: {output_path}")
        print()

    return 0

if __name__ == "__main__":
    main()