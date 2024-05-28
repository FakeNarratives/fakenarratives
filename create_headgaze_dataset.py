import argparse
import logging
import sys

import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
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
        "--delta_angle", type=int, default=2, help="delta for angle bins"
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
        gaze_faces = {"theta_x": {}, "theta_y": {}}
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

                left_eye = face["headgaze"][2]   ## In degrees
                right_eye = face["headgaze"][3]  ## In degrees

                # Average the absolute values and retain sign
                theta_x = (abs(left_eye[0]) + abs(right_eye[0])) / 2 * (1 if left_eye[0] + right_eye[0] >= 0 else -1)
                theta_y = (abs(left_eye[1]) + abs(right_eye[1])) / 2 * (1 if left_eye[1] + right_eye[1] >= 0 else -1)

                # discard faces with large angles for head gaze
                if abs(theta_x) > args.max_angle or abs(theta_y) > args.max_angle:
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

                # store face and gaze information
                for idx, angle in enumerate(["theta_x", "theta_y"]):
                    gaze_idx = str(
                        math.floor(abs([theta_x, theta_y][idx]) / args.delta_angle)
                    )
                    if gaze_idx not in gaze_faces[angle]:
                        gaze_faces[angle][gaze_idx] = []

                    logging.debug(f"{[theta_x, theta_y][idx]} -> {angle}/{gaze_idx}")

                    gaze_faces[angle][gaze_idx].append(face["id"])
                    face_lut[face["id"]] = {
                        "face_image": frame["frame"][y1:y2, x1:x2, :],
                        "gaze": [theta_x, theta_y],
                        "full_gaze": [left_eye, right_eye],
                        "time": face["time"],
                    }

        # sample images from gaze angles
        with open(os.path.join(args.output, f"{channelname}_{videoname}_face_angles_map.txt"), "w") as fw:
            for idx_theta_x in gaze_faces["theta_x"]:
                for idx_theta_y in gaze_faces["theta_y"]:

                    theta_x_faces = set(gaze_faces["theta_x"][idx_theta_x])
                    theta_y_faces = set(gaze_faces["theta_y"][idx_theta_y])

                    faces = theta_x_faces.intersection(theta_y_faces)
                    faces = list(faces)
                    num_faces = min(len(faces), args.faces_per_pose)

                    # TODO sort images by time before using linspace
                    sampled_idx = np.linspace(0, len(faces) - 1, num=num_faces)

                    dirname = pathlib.Path(f"{args.output}/{channelname}/{videoname}")
                    dirname.mkdir(parents=True, exist_ok=True)

                    for sidx in sampled_idx:
                        face_id = faces[round(sidx)]
                        face_image = face_lut[face_id]["face_image"]

                        fname = f"{idx_theta_x}_{idx_theta_y}_{face_id}.jpg"
                        outfile = dirname / fname

                        logging.debug(f"outfile: {outfile}")
                        imageio.imwrite(outfile, face_image)
                        fw.write(f"{outfile} {face_id} {face_lut[face_id]['gaze']} {face_lut[face_id]['full_gaze']}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
