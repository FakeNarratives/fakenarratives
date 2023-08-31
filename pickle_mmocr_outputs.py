import argparse
import logging
import numpy as np
import os
import pickle
import sys

from mmocr.ocr import MMOCR
from video_decoder import VideoBatcher, VideoDecoder


def mmocr_pkl(
    mmocr_outputs: list,
    times: list,
    args,
) -> dict:
    """Converts outputs from MMOCR to a pkl

    Args:
        mmocr_outputs (list<dict>): list of dicts containing the mmocr outputs per frame
        times (list): times for the mmocr outputs of length t
        args (Namespace): arguments the script has been executed with

    Returns:
        dict: dictionary ready to write in a .pkl
            y (list<dict>): list of dicts containing the mmocr outputs per frame
                rec_texts (list<str>): List (length n) of recognized texts
                rec_scores (list<float>): List (length n) of confidence scores for each recognized text
                det_polygons (list<list>): List (length n) containing the polygons for the detected text
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": mmocr_outputs,
        "time": times,
        "delta_time": 1 / args.fps,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # required

    # Required arguments
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--output", type=str, required=True, help="path to the output folder")

    # optional
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--fps", type=int, required=False, default=2, help="fps to process video")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=1280, help="max dimension of the video frames"
    )

    # mmocr models
    parser.add_argument("--det", type=str, required=False, default="DBPP_r50", help="detection model for mmocr")
    parser.add_argument(
        "--recog", type=str, required=False, default="ABINet_Vision", help="recognition model for mmocr"
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

    # init ocr model
    ocr = MMOCR(recog=args.recog, det=args.det)
    videos = args.videos

    # extract features for all videos
    for video_path in videos:
        logging.info(f"Processing video: {video_path}")
        vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=args.fps)
        vb = VideoBatcher(video_decoder=vd, batch_size=args.batch_size)

        times = []
        indices = []
        mmocr_outputs = []

        for i, batch in enumerate(vb):
            logging.info(f"Batch: {i}")
            images = np.split(batch["frame"], batch["frame"].shape[0], axis=0)

            # FIXME compactTV video crash for the first batch
            try:
                # predict event classes for images
                ocr_output = ocr.readtext([x.squeeze() for x in images], show=False)
            except Exception as e:
                logging.error(e)
                continue

            mmocr_outputs.extend(ocr_output)
            times.extend(batch["time"])
            indices.extend(batch["index"])

        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output, vidname)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "mmocr.pkl"), "wb") as f:
            output_dict = mmocr_pkl(
                mmocr_outputs=mmocr_outputs,
                times=times,
                args=args,
            )
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {os.path.join(output_dir, 'mmocr.pkl')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
