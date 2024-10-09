import os
import sys
import torch
import uuid
import time
import pickle
import logging
import argparse
import insightface
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
from insightface.app import FaceAnalysis
sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging

assert insightface.__version__ >= "0.3"

def parse_args():
    parser = argparse.ArgumentParser(description="insightface detection and facial feature extraction")
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
    parser.add_argument("--ctx", default=0, type=int, help="ctx id, <0 means using cpu")
    parser.add_argument("--det-size", default=640, type=int, help="detection size")
    parser.add_argument("--fps", type=int, default=25, help="frames per second")
    parser.add_argument("-w", "--workers", type=int, default=4, help="number of workers")
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1280,
        help="max dimension of the video frames"
    )
    return parser.parse_args()

def faceanalysis_pkl(faces: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]
    return {
        "y": faces,
        "args": args_dict,
    }

def get_faces(model: FaceAnalysis, vd: List[Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    faces = []
    for fid, frame in enumerate(tqdm(vd, desc="Processing frames")):
        time = fid / args.fps
        image = frame

        for face in model.get(image):
            x, y = round(max(0, face["bbox"][0])), round(max(0, face["bbox"][1]))
            w, h = round(face["bbox"][2] - x), round(face["bbox"][3] - y)

            image_height, image_width = image.shape[:2]

            faces.append({
                "id": str(uuid.uuid4()),
                "bbox": {
                    "x": float(x / image_width),
                    "y": float(y / image_height),
                    "w": float(w / image_width),
                    "h": float(h / image_height),
                },
                "kps": [[float(face["kps"][i, 0] / image_width), float(face["kps"][i, 1] / image_height)]
                        for i in range(face["kps"].shape[0])],
                "det_score": float(face["det_score"]),
                "time": time,
                "delta_time": 1 / args.fps,
                "frame": fid,
                "embedding": face["embedding"].tolist(),
                "landmark": face["landmark_3d_68"].tolist(),
            })

    return faces

def process_video(video_path: Path, output_dir: Path, model: FaceAnalysis, args: argparse.Namespace) -> Tuple[bool, str]:

    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, args.fps)
    logging.info(f"Video info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    args.frame_width = frame_width
    args.frame_height = frame_height

    faces = get_faces(model, vd, args)
    logging.info(f"Detected {len(faces)} faces in video")

    return faces
    

def process_videos(videos: List[str], pkl_dir: str, model: FaceAnalysis, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
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
        output_file = output_dir / "face_detection_insightface.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            faces = process_video(video_path, output_dir, model, args)

            with open(output_file, "wb") as f:
                output_dict = faceanalysis_pkl(faces, args)
                pickle.dump(output_dict, f)

            logging.info(f"Face detection output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main() -> None:
    args = parse_args()
    log_file = setup_logging("face_detection")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    model = FaceAnalysis(allowed_modules=["detection", "recognition", "landmark_3d_68"])
    model.prepare(ctx_id=args.ctx, det_size=(args.det_size, args.det_size))

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, model, args)

    # Log summary
    total = successful + failed
    logging.info(f"Face detection processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nFace detection processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()