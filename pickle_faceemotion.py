import argparse
import os
import pickle
import logging
import sys
from video_decoder import VideoDecoder

# 3rd party dependencies
import gdown
import numpy as np
import cv2
from tqdm import tqdm

# project dependencies
from deepface.commons import package_utils
# -------------------------------------------
# pylint: disable=line-too-long
# -------------------------------------------
# dependency configuration
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
# -------------------------------------------


def load_model(
    checkpoint_path: str,
) -> Sequential:
    """
    Consruct emotion model, download and load weights
    """

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

    # ----------------------------

    if os.path.isfile(checkpoint_path) != True:
        logging.info("facial_expression_model_weights.h5 will be downloaded...")
        url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"
        gdown.download(url, checkpoint_path, quiet=False)

    model.load_weights(checkpoint_path)

    return model

def preprocess_batch(face_imgs, target_size=(48, 48)):
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


def emotion_pkl(emotions: list, args) -> dict:
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]
    return {
        "y": emotions,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument(
        "--videos", nargs="+", type=str, required=True, help="path to the video files"
    )
    parser.add_argument(
        "--pkl_dir", type=str, required=True, help="path to the output folder"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="pretrained_models/facial_expression_model_weights.h5", help="path to checkpoint file",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--cpu", action="store_true", help="process on cpu")
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    # Create model and load fixed model parameters
    model = load_model(args.ckpt_path)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    batch_size = args.batch_size

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)
        face_detection_path = os.path.join(output_dir, "face_detection_insightface.pkl")

        if not os.path.isfile(face_detection_path):
            logging.error(f"\tMissing file: {face_detection_path}")
            continue

        with open(face_detection_path, "rb") as pklfile:
            face_content = pickle.load(pklfile)

        # get frames from video
        vd = VideoDecoder(video_path, max_dimension=args.max_dimension, fps=face_content["args"]["fps"])
        logging.info(f"\tVideo info: {vd._frames} frames, Original FPS: {vd._real_fps}, New FPS: {vd._fps}, Size: {vd._new_size[0]} x {vd._new_size[1]}")
        
        emotions = []
        face_batch = []
        face_info_batch = []

        for sample in tqdm(vd, desc="Processing frames"):
            image = sample["frame"]
            frame_index = sample["index"]
            
            # Filter faces for the current frame
            frame_faces = []
            for face in face_content["y"]:
                if face["frame"] == frame_index and face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05:
                    frame_faces.append(face)
            
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
                
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                face_batch.append(face_img)
                face_info_batch.append({"face_id": face["id"], "frame": frame_index, "time": face["time"]})

                if len(face_batch) == batch_size:
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

        os.makedirs(os.path.join(output_dir), exist_ok=True)
        output_path = os.path.join(output_dir, "face_emotions_deepface.pkl")
        with open(output_path, "wb") as f:
            output_dict = emotion_pkl(emotions, args)
            pickle.dump(output_dict, f)

        logging.info(f"\tFace emotions output written to: {output_path}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
