import argparse
import logging
import numpy as np
import os
import pickle
from scipy.cluster.hierarchy import fclusterdata
import sys


def faceclustering_pkl(
    ids: list,
    clusters: np.ndarray,
    args,
) -> dict:
    """Converts outputs from agglomerative face clustering to a pkl

    Args:
        ids (list): list (length n) of face ids
        clusters (list): list (length n) of corresponding cluster ids
        args (Namespace): arguments the script has been executed with

    Returns:
        dict: dictionary ready to write in a .pkl
            y (list<dict>): list of dicts containing the face clustering outputs
                ids (list): list (length n) of face ids
                clusters (list): list (length n) of corresponding cluster ids
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": {"ids": ids, "clusters": clusters},
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--output", type=str, required=True, help="path to the output folder")

    # optional
    parser.add_argument("--threshold", type=float, default=0.4, help="clustering threshold")
    parser.add_argument("--criterion", type=str, default="distance", help="clustering criterion")
    parser.add_argument("--metric", type=str, default="cosine", help="distance metric")

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

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

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

        # read face_analyis.pkl and store ids and embeddings
        with open(os.path.join(output_dir, "face_analysis.pkl"), "rb") as pklfile:
            content = pickle.load(pklfile)

        faces = content["output_data"][0]
        ids = []
        embeddings = []
        for face in faces["faces"]:
            ids.append(face["id"])
            embeddings.append(np.asarray(face["embedding"]))

        # cluster based on embeddings
        embeddings = np.asarray(embeddings)
        logging.debug(f"Perform clustering ... shape {embeddings.shape}")
        clusters = fclusterdata(X=embeddings, t=args.threshold, criterion=args.criterion, metric=args.metric)
        clusters = clusters.tolist()

        # return results
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "face_clustering.pkl"), "wb") as f:
            output_dict = faceclustering_pkl(ids, clusters, args)
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {os.path.join(output_dir, 'headpose_6DRepNet.pkl')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
