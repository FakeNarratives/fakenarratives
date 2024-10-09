import pickle
from tqdm import tqdm
import warnings
import os
import sys
import logging
from typing import Any
from pathlib import Path
import argparse
import numpy as np
from mmdet.apis import DetInferencer  # type: ignore
sys.path.append(".")
from video_utils import read_video_and_get_info


def parse_args():
    parser = argparse.ArgumentParser(description="Generates layout detections for frames sampled from video shots.")
    parser.add_argument("-m", "--model_dir", type=str, default="/nfs/data/fakenarratives/layout_analysis/saved_models/")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size for inference. Defaults to 1.")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )
    return parser.parse_args()


def sample_frames_from_shot(start_frame, end_frame, num_samples=5):
    """Sample frames evenly from a shot."""
    if end_frame - start_frame + 1 <= num_samples:
        return list(range(start_frame, end_frame + 1))
    step = (end_frame - start_frame) / (num_samples - 1)
    return [int(start_frame + i * step) for i in range(num_samples)]


def process_shot_results(shot_preds, class_names):
    """
    Process detection results for a shot to determine object presence.
    Apply a threshold of 0.8 on detection scores.
    """
    class_counts = {class_name: 0 for class_name in class_names}
    
    for pred in shot_preds:
        for label, score in zip(pred["labels"], pred["scores"]):
            if score >= 0.7:  # Apply the 0.7 threshold
                class_counts[class_names[label]] += 1
    
    return {class_name: count/5 for class_name, count in class_counts.items() if count/5 >= 0.5}

def process_video(video_path, model, args, shot_content, class_names):
    """
    Process a video by sampling frames from each shot and running object detection.
    """
    # Load video
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, 4, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    video_dict = {
        "github_repo": "https://github.com/open-mmlab/mmdetection",
        "commit_id": "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a",
        "parameters": "default", 
        "video_file": video_path,
        "output_data": []
    }

    for shot in tqdm(shot_content['output_data']['shots'], desc="Processing shots"):
        start_frame, end_frame = shot['start_frame'], shot['end_frame']
        sampled_frames = sample_frames_from_shot(start_frame, end_frame)
        shot_frames = [vr[frame_idx] for frame_idx in sampled_frames]
        
        shot_preds = model(shot_frames)["predictions"]
        
        objects_cnt_dict = process_shot_results(shot_preds, class_names) ## {"background": 2.0, "photo_moving": 1.0, "photo_still": 1.0}
        
        video_dict["output_data"].append({
            "start": shot["start"],
            "end": shot["end"],
            "layout_cnt": objects_cnt_dict
        })

    return video_dict

def unstack_array(array):
    """ unstacks an numpy array into a list of subarrays.

    Args:
        arr (np.ndarray): array with N+1 dimensions

    Returns:
        list[np.ndarray]: list of arrays with N dimensions
    """
    splitted_array = np.split(array, len(array))
    return [np.squeeze(subarray, 0) for subarray in splitted_array]


def main():
    args = parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    weights_path = os.path.join(args.model_dir, "co-detr_mixedtraining_2024-05-19_15-08/final_model.pth")
    inferencer = DetInferencer(weights=weights_path, show_progress=False)

    class_names = inferencer.cfg.test_dataloader.dataset.metainfo.classes
    print(f"\tClass names: {class_names}")

    for vi, video_path in enumerate(args.videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(args.videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")

        if not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot detection file in {shot_detection_path}")
            continue

        with open(shot_detection_path, "rb") as pklfile:
            shot_content = pickle.load(pklfile)

        logging.info(f"\tNumber of shots: {len(shot_content['output_data']['shots'])}")

        video_feat_dict = process_video(video_path, inferencer, args, shot_content, class_names)
        
        output_fp = os.path.join(output_dir, "layout_detection.pkl")
        with open(output_fp, "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)
        
        logging.info(f"\tSaved layout detections to {output_fp}")

if __name__ == "__main__":
    main()