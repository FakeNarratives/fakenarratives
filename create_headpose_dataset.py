import argparse
import logging
import sys

import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
from video_decoder import VideoDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument(
        "--videos",
        nargs="+",
        type=str,
        required=True,
        help="path to the video folder",
    )
    parser.add_argument("--pkl", type=str, required=True, help="path to the pkl folder")
    parser.add_argument(
        "--output", type=str, required=True, help="path to the output folder"
    )

    # optional
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1920,
        help="max dimension of the video frames",
    )
    parser.add_argument(
        "--delta_angle", type=int, default=5, help="delta for angle bins"
    )
    parser.add_argument(
        "--max_angle", type=int, default=20, help="delta for angle bins"
    )
    parser.add_argument(
        "--faces_per_pose", type=int, default=10, help="max number of faces per pose"
    )
    parser.add_argument(
        "--min_face_height", type=float, default=0.1, help="minimum size of the face"
    )
    parser.add_argument(
        "--min_face_det", type=float, default=0.85, help="minimum face detection score"
    )
    parser.add_argument(
        "--pad", type=float, default=0.1, help="minimum size of the face"
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

    # loop trough videos
    for video in args.videos:
        headpose_faces = {"pitch": {}, "yaw": {}, "roll": {}}
        face_lut = {}

        logging.info(video)
        videopath = pathlib.Path(video)
        videoname = videopath.stem
        channelname = videopath.parent.name

        pklpath = pathlib.Path(args.pkl)

        # read faces
        with open(pklpath / channelname / videoname / "face_analysis.pkl", "rb") as f:
            faces_data = pickle.load(f)

        faces = faces_data["faces"]
        fps = faces_data["plugins"][0]["parameters"]["fps"]

        # create dictionary including timestamps where faces are visible
        faces_dict = {}
        for face in faces:
            if face["time"] not in faces_dict:
                faces_dict[face["time"]] = []

            faces_dict[face["time"]].append(face)

        faces_times = np.asarray(list(faces_dict.keys()))

        # read video
        vd = VideoDecoder(path=video, max_dimension=args.max_dimension, fps=1)

        # loop trough video frames
        for frame in vd:
            time = frame["time"]
            logging.info(f"- time: {time}")

            delta_time = np.abs(time - faces_times)
            closest_face_time = faces_times[np.argmin(delta_time)]
            eps = 1 / fps

            # skip frames where no face is visible
            if np.abs(closest_face_time - time) > eps / 2:
                continue

            # loop trough visible faces
            for face in faces_dict[closest_face_time]:  # loop through face(s) in frame
                # discard small faces
                if face["bbox"]["h"] < args.min_face_height:
                    continue

                # discard faces with low detection score
                if face["bbox"]["det_score"] < args.min_face_det:
                    continue

                # discard faces with large angles for head poses
                if (
                    abs(face["headpose"][0]) >= args.max_angle
                    or abs(face["headpose"][1]) >= args.max_angle
                    or abs(face["headpose"][2]) >= args.max_angle
                ):
                    continue

                # scale bounding box to image resolution
                h, w, _ = frame["frame"].shape
                bbox = face["bbox"]

                pad_x = round(w * args.pad)
                pad_y = round(h * args.pad)

                y1 = max(0, round(h * bbox["y"]) - pad_y)
                y2 = min(h, round(h * (bbox["y"] + bbox["h"])) + pad_y)

                x1 = max(0, round(w * bbox["x"]) - pad_x)
                x2 = min(w, round(w * (bbox["x"] + bbox["w"])) + pad_x)

                # # get indices from the headpose for uniform sampling
                # idx_pitch = math.floor(abs(face["headpose"][0]) / args.delta_angle)
                # idx_yaw = math.floor(abs(face["headpose"][1]) / args.delta_angle)
                # idx_roll = math.floor(abs(face["headpose"][2]) / args.delta_angle)

                # store face and headpose information
                for idx, angle in enumerate(["pitch", "yaw", "roll"]):
                    headpose_idx = str(
                        math.floor(abs(face["headpose"][idx]) / args.delta_angle)
                    )
                    if headpose_idx not in headpose_faces[angle]:
                        headpose_faces[angle][headpose_idx] = []

                    logging.debug(f"{face['headpose'][idx]} -> {angle}/{headpose_idx}")

                    headpose_faces[angle][headpose_idx].append(face["id"])
                    face_lut[face["id"]] = {
                        "face_image": frame["frame"][y1:y2, x1:x2, :],
                        "headpose": face["headpose"],
                        "time": face["time"],
                    }

        # sample images from head poses
        for idx_pitch in headpose_faces["pitch"]:
            for idx_yaw in headpose_faces["yaw"]:
                for idx_roll in headpose_faces["roll"]:

                    pitch_faces = set(headpose_faces["pitch"][idx_pitch])
                    yaw_faces = set(headpose_faces["yaw"][idx_yaw])
                    roll_faces = set(headpose_faces["roll"][idx_roll])

                    faces = pitch_faces.intersection(yaw_faces, roll_faces)
                    faces = list(faces)
                    num_faces = min(len(faces), args.faces_per_pose)

                    # TODO sort images by time before using linspace
                    sampled_idx = np.linspace(0, len(faces) - 1, num=num_faces)

                    dirname = pathlib.Path(f"{args.output}/{channelname}/{videoname}")
                    dirname.mkdir(parents=True, exist_ok=True)

                    for sidx in sampled_idx:
                        face_id = faces[round(sidx)]
                        face_image = face_lut[face_id]["face_image"]

                        fname = f"{idx_pitch}_{idx_yaw}_{idx_roll}_{face_id}.jpg"
                        outfile = dirname / fname

                        # logging.debug(f"{face_lut[face_id]['time']}")
                        logging.debug(f"outfile: {outfile}")
                        imageio.imwrite(outfile, face_image)

    return 0


if __name__ == "__main__":
    sys.exit(main())
