import argparse
import logging
import numpy as np
import os
import pickle
import sys
from tqdm import tqdm

from sixdrepnet import SixDRepNet
from video_utils import read_video_and_get_info


def headpose_pkl(
    headposes: list,
    args,
) -> dict:
    """Converts outputs from 6DRepNet for head pose estimation to a pkl

    Args:
        headposes (list<dict>): list (length n) containing the id and headpose of a face
        args (Namespace): arguments the script has been executed with

    Returns:
        dict: dictionary ready to write in a .pkl
            y (list<dict>): list of dicts containing the headpose outputs per frame
                id (dict): face id according to the face_analysis.pkl input file
                headpose (list<dict>): list (length n) of head poses including
                    pitch (float): pitch angle of the face (in degree)
                    yaw (float): yaw angle of the face (in degree)
                    roll (float): roll angle of the face (in degree)
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": headposes,
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

    # optional
    parser.add_argument(
        "--batch_size", type=int, required=False, default=8, help="batch size"
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

    # Create model
    device = -1 if args.cpu else 0
    model = SixDRepNet(
        gpu_id=device, dict_path="6DRepNet/model/6DRepNet_300W_LP_AFLW2000.pth"
    )

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")

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
        vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, face_content["args"]["fps"])
        logging.info(f"\tVideo info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
        assert frame_width == face_content["args"]["frame_width"]
        assert frame_height == face_content["args"]["frame_height"]

        headposes = []
        for frame_index, image in enumerate(tqdm(vd, desc="Processing frames")):
            # Filter faces for the current frame
            frame_faces = []
            for face in face_content["y"]:
                if face["frame"] == frame_index and face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05:
                    frame_faces.append(face)
            
            if not frame_faces:
                continue

            # Rescale normalized bounding boxes and keypoints to the original image size
            for face in frame_faces:
                y1 = round(frame_height * face["bbox"]["y"])
                y2 = round(frame_height * (face["bbox"]["y"] + face["bbox"]["h"]))

                x1 = round(frame_width * face["bbox"]["x"])
                x2 = round(frame_width * (face["bbox"]["x"] + face["bbox"]["w"]))

                face_image = image[y1:y2, x1:x2, :]
                pitch, yaw, roll = model.predict(face_image)

                headposes.append(
                    {
                        "face_id": face["id"],
                        "frame": frame_index,
                        "time": face["time"],
                        "headpose": {"pitch": pitch[0], "yaw": yaw[0], "roll": roll[0]},
                    }
                )

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "headpose_6DRepNet.pkl"), "wb") as f:
            output_dict = headpose_pkl(headposes, args)
            pickle.dump(output_dict, f)

        logging.info(
            f"\tOutput written to: {os.path.join(output_dir, 'headpose_6DRepNet.pkl')}"
        )

        logging.debug(len(headposes))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
