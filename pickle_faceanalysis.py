import argparse
import logging
import numpy as np
import os
import pickle
import sys

from graph_visualization.utils import read_pkl


def faceanalysis_pkl(
    faces: list,
) -> dict:
    """Converts outputs from 6DRepNet for head pose estimation to a pkl

    Args:
        faces (<dict>):

    Returns:
        dict: dictionary ready to write in a .pkl
            faces (list<dict>): list of dicts containing extracted information of
                                each face detected in the video
                id (dict): face id according to the face_analysis.pkl input file
                time (float): time of the frame the face is visible in (in seconds)
                delta_time (float): frame duration the face is visible (in seconds)
                bbox (dict): bounding box information of the face
                    x (float): relative x position of the face in the frame
                    y (float): relative y position of the face in the frame
                    w (float): relative width of the face in the frame
                    h (float): relative height position of the face in the frame
                    det_score (float): detection score of the face detector
                kps (list<float>): list of five two-dimensional facial keypoints
                emotion (np.ndarray): np.ndarray containing the probabilities of 7 facial emotions
                    Angry (float)
                    Disgust (float)
                    Fear (float)
                    Happy (float)
                    Sad (float)
                    Surprise (float)
                    Neutral (float)
                headpose (np.array): np.ndarray containing three headpose angles
                    pitch (float): pitch angle of the face (in degree)
                    yaw (float): yaw angle of the face (in degree)
                    roll (float): roll angle of the face (in degree)
                cluster_id (int): cluster id of the face
            plugins (dict): dictionary containing the parameters of the performed plugins
    """

    faceanalysis_dict = {"faces": []}
    for key, val in faces.items():
        if key == "output_data":
            continue
        faceanalysis_dict[key] = val

    for face in faces["output_data"][0]["faces"]:
        face_dict = {}
        for key, val in face.items():
            if key == "embedding":
                continue
            face_dict[key] = val

        faceanalysis_dict["faces"].append(face_dict)

    return faceanalysis_dict


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument(
        "--input", type=str, required=True, help="path to the pkl folder"
    )

    # optional
    parser.add_argument("--debug", action="store_true", help="debug output")

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

    # read face clustering results
    face_clusters = read_pkl(os.path.join(args.input, "face_clustering.pkl"))
    cluster_lut = {}
    for i, face_id in enumerate(face_clusters["y"]["ids"]):
        cluster_id = face_clusters["y"]["clusters"][i]
        cluster_lut[face_id] = cluster_id

    # read head pose results
    headposes = read_pkl(os.path.join(args.input, "headpose_6DRepNet.pkl"))
    headpose_lut = {}
    for entry in headposes["y"]:
        headpose_lut[entry["id"]] = np.asarray(
            [
                entry["headpose"]["pitch"],
                entry["headpose"]["yaw"],
                entry["headpose"]["roll"],
            ]
        )

    # read faces and store all information
    faces_data = read_pkl(os.path.join(args.input, "face_analysis_tibava.pkl"))
    for face in faces_data["output_data"][0]["faces"]:
        face["headpose"] = headpose_lut[face["id"]]
        face["cluster_id"] = cluster_lut[face["id"]]

        # print(face)
        break

    # write pkl file
    with open(os.path.join(args.input, "face_analysis.pkl"), "wb") as f:
        output_dict = faceanalysis_pkl(faces_data)
        pickle.dump(output_dict, f)

    logging.info(f"Output written to: {os.path.join(args.input, 'face_analysis.pkl')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
