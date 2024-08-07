import argparse
import os.path as osp
import os
import pickle
import sys
import cv2
import numpy as np
from tqdm import tqdm
import logging

sys.path.append("3DGazeNet/")
from models import GazeModel
from handler import *

from decord import VideoReader, cpu

def headgaze_pkl(headgazes: list, args) -> dict:
    """Converts outputs from 3DGazeNet for gaze estimation to a pkl"""
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": headgazes,
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
        "--ckpt_path", type=str, default="3DGazeNet/assets/latest_a.ckpt", help="path to checkpoint file",
    )
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

    # load 3DGazeNet model
    handler = GazeHandler(args.ckpt_path)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")

        # setup output dir
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        if not os.path.isfile(os.path.join(output_dir, "face_detection_insightface.pkl")):
            logging.error(
                f"Missing file: {os.path.join(output_dir, 'face_detection_insightface.pkl')}"
            )
            continue

        # read faceanalyis.pkl and store face bboxes
        with open(os.path.join(output_dir, "face_detection_insightface.pkl"), "rb") as pklfile:
            face_content = pickle.load(pklfile)

        # get frames from video
        vd, new_width, new_height, fps = read_video_and_get_info(video_path, args)
        args.fps = fps
        logging.info(f"Video info: {len(vd)} frames, {fps} FPS, {new_width} x {new_height}")

        headgazes = []
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

            logging.debug(f"{vidname}: Processing frame {frame_index}")

            # Rescale normalized bounding boxes and keypoints to the original image size
            for face in frame_faces:
                face["bbox"]["w"] *= image.shape[1]
                face["bbox"]["h"] *= image.shape[0]
                face["bbox"]["x"] *= image.shape[1]
                face["bbox"]["y"] *= image.shape[0]
                face["kps"] = np.array(face["kps"]) * [image.shape[1], image.shape[0]]
            
            results = handler.get(frame_faces, image)
            
            for face, result in zip(frame_faces, results):
                gaze_pred_l_rad, gaze_pred_r_rad = handler.get_gaze(eye_kps=result[2])
                gaze_pred_l_deg, gaze_pred_r_deg = np.degrees(gaze_pred_l_rad), np.degrees(gaze_pred_r_rad)
                
                headgazes.append({
                    "face_id": face["id"],
                    "frame": frame_index,
                    "time": face["time"],
                    "headgaze": {
                        "left_gaze_rad": gaze_pred_l_rad.tolist(),
                        "right_gaze_rad": gaze_pred_r_rad.tolist(),
                        "left_gaze_deg": gaze_pred_l_deg.tolist(),
                        "right_gaze_deg": gaze_pred_r_deg.tolist(),
                    },
                })

        os.makedirs(os.path.join(output_dir), exist_ok=True)
        output_path = os.path.join(output_dir, "headgaze_3DGazeNet.pkl")
        with open(output_path, "wb") as f:
            output_dict = headgaze_pkl(headgazes, args)
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {output_path}")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
