import os
import pickle
from tqdm import tqdm
import numpy as np
from transnetv2 import TransNetV2
import argparse
import logging
import sys
import time
sys.path.append(".")
from video_utils import read_video_and_get_info

def parse_args():
    parser = argparse.ArgumentParser(description="Performs shot detection in a video using TransNetV2.")
    parser.add_argument(
        "--videos", nargs="+", type=str, required=True, help="path to the video files"
    )
    parser.add_argument(
        "--pkl_dir", type=str, required=True, help="path to the output folder"
    )
    parser.add_argument("--fps", type=int, default=25, help="frames per second")
    parser.add_argument("--debug", action="store_true", help="debug output")
    args = parser.parse_args()
    return args


def process_video(video_path, model, args):
    """
    Use TransNetV2 model to perform shot detection on the video.

    Args:
        video_path (str): path to video
        output_path (str): path to save output
        model (TransNetV2): TransNetV2 model

    Returns:
        shot_dict (dict): dictionary containing shot detection results
    """
    shot_dict = {
        "github_repo": "https://github.com/soCzech/TransNetV2",
        "commit_id": "85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed", 
        "parameters": "default",
        "video_file": video_path,
        "output_data": {"shots": []}
    }

    # get frames from video
    vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.fps)
    logging.info(f"\tVideo info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    all_frames = np.array([vd[i] for i in range(len(vd))])
    logging.info(f"\tAll Frames: {all_frames.shape}")

    single_frame_predictions, _ = model.predict_frames(all_frames)

    # Process predictions
    shots = model.predictions_to_scenes(single_frame_predictions)

    for shot in shots:
        start_frame, end_frame = shot
        start_time = start_frame / args.fps
        end_time = end_frame / args.fps

        shot_dict["output_data"]["shots"].append({
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "start": start_time,
            "end": end_time
        })

    return shot_dict

def main():
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

    model = get_model()

    videos = args.videos
    for vi, video_path in enumerate(videos):
        start = time.time()

        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")

        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file_path = os.path.join(output_dir, "transnet_shotdetection.pkl")

        shot_dict = process_video(video_path, model, args)

        with open(output_file_path, "wb") as output_file:
            pickle.dump(shot_dict, output_file)

        logging.info(f"\tShot Prediction output saved at: {output_file_path}")

        logging.info(f"\tTime taken: {time.time() - start:.2f} seconds")

        print()

        break

if __name__ == "__main__":
    main()