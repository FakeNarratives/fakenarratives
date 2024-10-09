import os
import sys
import torch
import pickle
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoImageProcessor, VideoMAEModel
from transformers import XCLIPProcessor, XCLIPModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neibhboring shots in a video.")
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-m", "--model", type=str, default="kinetics-vmae", help="kinetics-vmae | ssv2-vmae | kinetics-xclip")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="number of workers to use for data loading")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=640,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


def get_model(model_type, device):
    if model_type == "kinetics-vmae":
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
    elif model_type == "ssv2-vmae":
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
    elif model_type == "kinetics-xclip":
        model = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14-kinetics-600")
        processor = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14-kinetics-600")
        
    model.eval()
    model.to(device)

    print("\nModel Loaded: ", model_type,"\n")

    return model, processor


def get_shot_frames(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start_frame"], shots[index]["end_frame"]
    return None, None


def sample_frame_indices(clip_len, indices):
    if len(indices) <= clip_len:
        return indices + [indices[-1]] * (clip_len - len(indices))
    else:
        step_size = max(1, len(indices) // clip_len)
        return [indices[i] for i in range(0, len(indices), step_size)][:clip_len] 


def process_frames(frames, processor, model_type):
    if 'vmae' in model_type:
        inputs = processor(frames, return_tensors="pt")
    else:
        inputs = processor(videos=frames, return_tensors="pt")

    return inputs


def compute_similarity(ref_feat, query_video, model_type, model, processor, device):
    query_inputs = process_frames(query_video, processor, model_type).to(device)
    
    with torch.no_grad():
        if "vmae" in model_type:
            outputs = model(**query_inputs).last_hidden_state.squeeze(0)
            query_feat = outputs[0].unsqueeze(0).cpu().numpy()
        else:
            query_feat = model.get_video_features(**query_inputs).cpu().numpy()
        
        similarity = cosine_similarity(ref_feat, query_feat)[0][0]

    return similarity


def process_video(video_path, shot_dict, model_type, model, processor, args, device="cuda"):
    """
    Take VideoMAE model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        shot_dict (dict): dictionary containing shot information
        model_type (str): kinetics-vmae | ssv2-vmae | kinetics-xclip
        model (AutoModel): VideoMAE model
        processor (AutoProcessor): VideoMAE processor
        args (Namespace): arguments the script has been executed with
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing shot similarity between two neibhboring shots from the reference shot in a video.
        As:
        {
            "github_repo": "",
            "commit_id": "",
            "parameters": "default",
            "video_file": video_path,
            "output_data": [
                {
                    "shot": {
                        "start": 0.0,
                        "end": 1.0
                    },
                    "prev_1": 0.0,  ## Video level similarity score over 16/8 frames
                    "prev_2": 0.0,
                    "next_1": 0.0,
                    "next_2": 0.0 
                }
            ]
        }
    """

    video_feat_dict = {"github_repo": "", "commit_id": "", "parameters": "default", "video_file": video_path, "output_data": []}
    
    if model_type == "kinetics-vmae":
        video_feat_dict["github_repo"] = "https://huggingface.co/MCG-NJU/videomae-large-finetuned-kinetics"
        video_feat_dict["commit_id"] = "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"
    elif model_type == "kinetics-ssv2":
        video_feat_dict["github_repo"] = "https://huggingface.co/MCG-NJU/videomae-base-finetuned-ssv2"
        video_feat_dict["commit_id"] = "e3f23b7a9bcc6f6b0bbf44da097c0f94548c6ec5"
    elif model_type == "kinetics-xclip":
        video_feat_dict["github_repo"] = "https://huggingface.co/microsoft/xclip-large-patch14-kinetics-600"
        video_feat_dict["commit_id"] = "0fed3494a823681faf820651bda82264853ccf78"


    # Load video
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.num_workers, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    total_frames = len(vr)

    clip_len = 16 if "vmae" in model_type else 8

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        ## For each shot and reference shot - cls tag similarity from last hidden state
        similarities = {"shot": ref_shot, "prev_1": 0, "prev_2": 0, "next_1": 0, "next_2": 0}

        ## Compute reference shot feats
        ref_indices = [k for k in range(ref_shot["start_frame"], min(ref_shot["end_frame"] + 1, total_frames))]
        ref_indices = sample_frame_indices(clip_len, ref_indices)
        ref_frames = vr.get_batch(ref_indices)
        ref_inputs = process_frames(ref_frames, processor, model_type).to(device)
        with torch.no_grad():
            if "vmae" in model_type:
                outputs = model(**ref_inputs).last_hidden_state.squeeze(0)
                ref_feat = outputs[0].unsqueeze(0).cpu().numpy()
            else:
                ref_feat = model.get_video_features(**ref_inputs).cpu().numpy()

        # Time frames for the current and neighboring shots
        shot_frames = [get_shot_frames(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        # Extract and compute for each neighboring shot
        for j, (start_frame, end_frame) in enumerate(shot_frames):
            if start_frame is not None and end_frame is not None:
                query_indices = [k for k in range(start_frame, min(end_frame + 1, total_frames))]
                ## Sample 16 frames from the reference shot
                query_indices = sample_frame_indices(clip_len, query_indices)
                query_frames = vr.get_batch(query_indices)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feat, list(query_frames), model_type, model, processor, device)


        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict



def main():
    args = parse_args()

    set_seeds(42)

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_model(args.model, device)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")

        if not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot detection file in {shot_detection_path}")
            continue

        with open(shot_detection_path, "rb") as pklfile:
            shot_content = pickle.load(pklfile)

        logging.info(f"\tNumber of shots: {len(shot_content['output_data']['shots'])}")

        video_feat_dict = process_video(video_path, shot_content, args.model, model, processor, args, device)

        with open(os.path.join(output_dir, "%s_action_shot_similarity.pkl"%(args.model)), "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
