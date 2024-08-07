import argparse
import os.path as osp
import os
import pickle
import logging
import sys
from decord import VideoReader, cpu

# 3rd party dependencies
import gdown
import numpy as np
import cv2
from tqdm import tqdm

# project dependencies
from deepface.commons import package_utils, folder_utils
from deepface.models.Demography import Demography
from deepface.commons.logger import Logger
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
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
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


def read_video_and_get_info(video_path, args):
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ## Make longer side of the frame equal to max_dimension  if it is greater than max_dimension
    if original_width > args.max_dimension:
        new_width = args.max_dimension
        new_height = int(original_height * new_width / original_width)
    else:
        new_width = original_width
        new_height = original_height

    vr = VideoReader(video_path, num_threads=12, ctx=cpu(0), width=new_width, height=new_height)
    fps = vr.get_avg_fps()
    
    return vr, new_width, new_height, fps


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

    batch_size = args.batch_size  # Adjust this based on your GPU memory

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)
        face_detection_path = os.path.join(output_dir, "face_detection_insightface.pkl")

        if not os.path.isfile(face_detection_path):
            logging.error(f"Missing file: {face_detection_path}")
            continue

        with open(face_detection_path, "rb") as pklfile:
            face_content = pickle.load(pklfile)

        # vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=face_content["args"]["fps"])
        vd, new_width, new_height, fps = read_video_and_get_info(video_path, args)
        args.fps = fps
        logging.info(f"Video info: {len(vd)} frames, {fps} FPS, {new_width} x {new_height}")
        
        emotions = []
        face_batch = []
        face_info_batch = []

        for fid, frame in enumerate(tqdm(vd, desc="Processing frames")):
            # image = frame["frame"]
            # frame_index = frame["index"]
            image = frame.asnumpy()
            frame_index = fid
            
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
                    emotion_predictions = model.predict(preprocessed_faces)
                    
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
            emotion_predictions = model.predict(preprocessed_faces)
            
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

        logging.info(f"Output written to: {output_path}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
