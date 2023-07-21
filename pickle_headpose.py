import argparse
import logging
import os
import pickle
import sys

from sixdrepnet import SixDRepNet
from video_decoder import VideoDecoder


def headpose_pkl(
    headposes: list,
    times: list,
    args,
) -> dict:
    """Converts outputs from 6DRepNet for head pose estimation to a pkl

    Args:
        headposes (list<dict>): list (length n) containing the bbox and headpose of a face
        times (list): time values for each headpose output (length n)
        args (Namespace): arguments the script has been executed with

    Returns:
        dict: dictionary ready to write in a .pkl
            y (list<dict>): list of dicts containing the mmocr outputs per frame
                bbox (dict): dictionary containing the bounding box for the face
                    x (float): x-coordinate of the face normalized by the image width
                    y (float): y-coordinate of the face normalized by the image height
                    w (float): width of the face normalized by the image width
                    h (float): height of the face normalized by the image height
                    det_score (float): likelihood of the bounding box beeing a face

                headpose (list<dict>): list (length n) of head poses including
                    pitch (float): pitch angle of the face (in degree)
                    yaw (float): yaw angle of the face (in degree)
                    roll (float): roll angle of the face (in degree)

            time (list): t time values (length t)
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": headposes,
        "time": times,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--output", type=str, required=True, help="path to the output folder")

    # optional
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--cpu", action="store_true", help="process on cpu")
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=1920, help="max dimension of the video frames"
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

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # Create model
    device = -1 if args.cpu else 0
    model = SixDRepNet(gpu_id=device, dict_path="6DRepNet/model/6DRepNet_300W_LP_AFLW2000.pth")

    # loop trough input videos
    videos = args.videos
    for video_path in videos:
        logging.info(f"Processing video: {video_path}")

        # setup output dir
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output, vidname)

        if not os.path.isfile(os.path.join(output_dir, "face_analysis.pkl")):
            logging.error(f"Missing file: {os.path.join(output_dir, 'face_analysis.pkl')}")
            continue

        # read faceanalyis.pkl and store face bboxes
        with open(os.path.join(output_dir, "face_analysis.pkl"), "rb") as pklfile:
            content = pickle.load(pklfile)

        fps = content["plugins"][0]["parameters"]["fps"]
        faces = content["output_data"][0]

        faces_dict = {}
        for face in faces["faces"]:
            if face["time"] not in faces_dict:
                faces_dict[face["time"]] = []

            faces_dict[face["time"]].append(face)

        # get frames from video
        vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=fps)

        times = []
        headposes = []
        for frame in vd:
            time = frame["time"]
            image = frame["frame"]

            if time not in faces_dict:  # no face in frame
                continue

            logging.debug(f"{vidname}: Processing frame {frame['index']}")

            for face in faces_dict[time]:  # loop through face(s) in frame
                h, w, _ = image.shape

                y1 = round(h * face["bbox"]["y"])
                y2 = round(h * (face["bbox"]["y"] + face["bbox"]["h"]))

                x1 = round(w * face["bbox"]["x"])
                x2 = round(w * (face["bbox"]["x"] + face["bbox"]["w"]))

                face_image = image[y1:y2, x1:x2, :]
                pitch, yaw, roll = model.predict(face_image)

                headposes.append(
                    {"bbox": face["bbox"], "headpose": {"pitch": pitch[0], "yaw": yaw[0], "roll": roll[0]}}
                )
                times.append(time)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "headpose_6DRepNet.pkl"), "wb") as f:
            output_dict = headpose_pkl(headposes, times, args)
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {os.path.join(output_dir, 'headpose_6DRepNet.pkl')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
