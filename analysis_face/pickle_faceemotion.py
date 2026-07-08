import os
import sys
import time
import pickle
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import gdown
from deepface.commons import package_utils
sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds, setup_logging

tf_version = package_utils.get_tf_major_version()

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
else:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion detection using DeepFace")
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
        "--ckpt_path",
        type=str,
        default="pretrained_models/facial_expression_model_weights.h5",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=1280,
        help="Max dimension of the video frames"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of workers")
    return parser.parse_args()

def load_model(checkpoint_path: str) -> Sequential:
    num_classes = 7
    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    if not os.path.isfile(checkpoint_path):
        logging.info("facial_expression_model_weights.h5 will be downloaded...")
        url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"
        gdown.download(url, checkpoint_path, quiet=False)

    model.load_weights(checkpoint_path)
    return model

def preprocess_batch(face_imgs: List[np.ndarray], target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    processed_faces = []
    for img in face_imgs:
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)

            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), "constant")

        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)

        img_pixels = img.astype(np.float32) / 255.0
        processed_faces.append(img_pixels)

    return np.array(processed_faces)

def emotion_pkl(emotions: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]
    return {
        "y": emotions,
        "args": args_dict,
    }

def process_video(video_path: Path, output_dir: Path, model: Sequential, args: argparse.Namespace) -> List[Dict[str, Any]]:
    face_detection_path = output_dir / "face_detection_insightface.pkl"
    
    if not face_detection_path.exists():
        raise FileNotFoundError(f"Missing file: {face_detection_path}")

    with open(face_detection_path, "rb") as pklfile:
        face_content = pickle.load(pklfile)

    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(str(video_path), args, args.workers, face_content["args"]["fps"])
    logging.info(f"Video info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    
    assert frame_width == face_content["args"]["frame_width"]
    assert frame_height == face_content["args"]["frame_height"]

    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotions = []
    face_batch = []
    face_info_batch = []

    for frame_index, image in enumerate(tqdm(vd, desc="Processing frames")):
        frame_faces = [face for face in face_content["y"] if face["frame"] == frame_index and face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05]
        
        if not frame_faces:
            continue

        for face in frame_faces:
            x = int(face["bbox"]["x"] * image.shape[1])
            y = int(face["bbox"]["y"] * image.shape[0])
            w = int(face["bbox"]["w"] * image.shape[1])
            h = int(face["bbox"]["h"] * image.shape[0])
            
            face_img = image[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            
            face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            face_batch.append(face_img)
            face_info_batch.append({"face_id": face["id"], "frame": frame_index, "time": face["time"]})

            if len(face_batch) == args.batch_size:
                preprocessed_faces = preprocess_batch(face_batch)
                emotion_predictions = model.predict(preprocessed_faces, verbose=0)
                
                for face_info, prediction in zip(face_info_batch, emotion_predictions):
                    emotions.append({
                        "face_id": face_info["face_id"],
                        "frame": face_info["frame"],
                        "time": face_info["time"],
                        "emotions": {label: float(pred) for label, pred in zip(emotion_labels, prediction)}
                    })
                
                face_batch = []
                face_info_batch = []

    # Process any remaining faces
    if face_batch:
        preprocessed_faces = preprocess_batch(face_batch)
        emotion_predictions = model.predict(preprocessed_faces, verbose=0)
        
        for face_info, prediction in zip(face_info_batch, emotion_predictions):
            emotions.append({
                "face_id": face_info["face_id"],
                "frame": face_info["frame"],
                "time": face_info["time"],
                "emotions": {label: float(pred) for label, pred in zip(emotion_labels, prediction)}
            })

    logging.info(f"Detected {len(emotions)} emotions in video")
    return emotions

def process_videos(videos: List[str], pkl_dir: str, model: Sequential, args: argparse.Namespace) -> Tuple[int, int, List[str]]:
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
        output_file = output_dir / "face_emotions_deepface.pkl"
        
        if output_file.exists() and not args.rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        start = time.time()
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            
            emotions = process_video(video_path, output_dir, model, args)

            with open(output_file, "wb") as f:
                output_dict = emotion_pkl(emotions, args)
                pickle.dump(output_dict, f)

            logging.info(f"Face emotions output saved to: [{vi+1}/{len(videos)}]: {output_file}")
    
            successful += 1
            logging.info(f"Time taken: {time.time() - start:.2f} seconds")
        except Exception as e:
            failed += 1
            failed_videos.append(str(video_path))
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")

    return successful, failed, failed_videos

def main():
    args = parse_args()
    log_file = setup_logging("face_emotions")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    model = load_model(args.ckpt_path)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, model, args)

    # Log summary
    total = successful + failed
    logging.info(f"Face emotion detection processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nFace emotion detection processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()