import argparse
import logging
import numpy as np
import os
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Combine face analysis results")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("--debug", action="store_true", help="debug output")
    return parser.parse_args()

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def combine_face_analysis(features_dir):
    # Load all individual analysis results
    face_detection = load_pickle(os.path.join(features_dir, "face_detection_insightface.pkl"))
    face_tracking = load_pickle(os.path.join(features_dir, "face_tracking.pkl"))
    face_clustering = load_pickle(os.path.join(features_dir, "face_clustering.pkl"))
    headgaze = load_pickle(os.path.join(features_dir, "headgaze_3DGazeNet.pkl"))
    asd_results = load_pickle(os.path.join(features_dir, "asd_light-asd.pkl"))
    emotions = load_pickle(os.path.join(features_dir, "face_emotions_deepface.pkl"))

    # Create mappings for quick lookups
    face_id_to_track_id = {}
    for track in face_tracking['tracks']:
        for frame in track['frames']:
            face_id_to_track_id[frame['face_id']] = track['track_id']


    clustering_map = {item['face_id']: item['cluster_id'] for item in face_clustering['y']}
    headgaze_map = {item['face_id']: item['headgaze'] for item in headgaze['y']}
    asd_map = {track['track_id']: track for track in asd_results['tracks']}
    emotions_map = {item['face_id']: item['emotions'] for item in emotions['y']}

    combined_faces = []

    for face in face_detection['y']:
        face_id = face['id']
        track_id = face_id_to_track_id.get(face_id)
        
        asd_info = asd_map.get(track_id, {})
        
        combined_face = {
            "face_id": face_id,
            "time": face['time'],
            "frame": face['frame'],
            "bbox": face['bbox'],
            "kpss": face['kps'],
            "emb": np.array(face['embedding']),
            "track_id": track_id,
            "cluster_id": clustering_map.get(face_id),
            "gaze": headgaze_map.get(face_id, {}),
            "speaking": asd_info.get('is_speaking', False),
            "speaking_ratio": asd_info.get('speaking_ratio', 0),
            "speaking_frames": asd_info.get('speaking_frames', 0),
            "speaking_scores": asd_info.get('smoothed_scores', []),
            "emotions": emotions_map.get(face_id, {})
        }
        
        combined_faces.append(combined_face)

    # Save the combined results
    os.makedirs(features_dir, exist_ok=True)
    save_pickle({"faces": combined_faces, "args": face_detection['args']}, 
                os.path.join(features_dir, "face_analysis.pkl"))

    logger.info(f"\tCombined face analysis saved to {os.path.join(features_dir, 'face_analysis.pkl')}")
    logger.info(f"\tTotal faces: {len(combined_faces)}")
    logger.info(f"\tSpeaking faces: {sum(1 for face in combined_faces if face['speaking'])}")

    print()


def main():
    args = parse_args()
    
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        input_dir = os.path.join(args.pkl_dir, vidname)
        
        combine_face_analysis(input_dir)

if __name__ == "__main__":
    main()