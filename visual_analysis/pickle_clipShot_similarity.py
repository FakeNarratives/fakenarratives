import os
import torch
import pickle
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neibhboring shots in a video.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def get_model(device):
    ## Using Google's SigLIP model - The best model for the task
    model = AutoModel.from_pretrained("google/siglip-large-patch16-256")
    processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")
    model.eval()
    model.to(device)
    return model, processor


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:# print(image_features.shape, text_features.shape)
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_shot_time(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start"], shots[index]["end"]
    return None, None

def extract_frames(vr, start_time, end_time, fps, subsample_rate=2):
    step_size = max(1, int(fps // subsample_rate))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frames = vr.get_batch(range(start_frame, end_frame + 1))
    frames = frames[::step_size]
    return frames


def compute_similarity(ref_feats, query_frames, model, processor, device):
    query_inputs = processor(images=query_frames, return_tensors="pt")
    query_inputs.to(device)

    with torch.no_grad():
        query_feats = model.get_image_features(**query_inputs).cpu().numpy()
        similarity = cosine_similarity(ref_feats, query_feats)

    return [np.mean(similarity), np.median(similarity), np.max(similarity)]


def process_video(video_path, output_path, model, processor, device="cuda"):
    """
    Take SigLIP model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        model (AutoModel): SigLIP model
        processor (AutoProcessor): SigLIP processor
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing shot similarity between two neibhboring shots from the reference shot in a video.
        As:
        {
            "github_repo": "https://huggingface.co/google/siglip-large-patch16-256",
            "commit_id": "3bae8ca81b490faa0803ba07cf45159cec2f2ef4",
            "parameters": "default",
            "video_file": video_path,
            "output_data": [
                {
                    "shot": {
                        "start": 0.0,
                        "end": 1.0
                    },
                    "prev_1": [0.0, 0.0, 0.0],  ## [mean, median, max] similarity scores
                    "prev_2": [0.0, 0.0, 0.0],
                    "next_1": [0.0, 0.0, 0.0],
                    "next_2": [0.0, 0.0, 0.0]
                }
            ]
        }
    """

    video_feat_dict = {"github_repo": "https://huggingface.co/google/siglip-large-patch16-256",
                        "commit_id": "3bae8ca81b490faa0803ba07cf45159cec2f2ef4",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": []
                        }


    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))

    # Load the video
    vr = VideoReader(video_path+".mp4", num_threads=4, ctx=cpu(0))
    fps = vr.get_avg_fps()

    similarities = []
    for i, ref_shot in enumerate(shot_dict["output_data"]["shots"]):
        ## For each shot and reference shot - have [mean, median and max] similarity scores
        similarities = {"shot": ref_shot, "prev_1": [0, 0, 0], "prev_2": [0, 0, 0], "next_1": [0, 0, 0], "next_2": [0, 0, 0]}

        ## Compute reference shot feats
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        ref_frames = extract_frames(vr, sr_time, er_time, fps, subsample_rate=2)
        ref_inputs = processor(images=ref_frames, return_tensors="pt").to(device)
        with torch.no_grad():
            ref_feats = model.get_image_features(**ref_inputs).cpu().numpy()

        # Time frames for the current and neighboring shots
        shot_times = [get_shot_time(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        # Extract and compute for each neighboring shot
        for j, (start_time, end_time) in enumerate(shot_times):
            if start_time is not None and end_time is not None:
                query_frames = extract_frames(vr, start_time, end_time, fps, subsample_rate=2)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feats, query_frames, model, processor, device)

        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    model, processor = get_model(device)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video {i+1}/{len(input_paths)}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "siglip_shot_similarity.pkl")):
            video_feat_dict = process_video(input_path, out_loc, model, processor, device)

            with open(os.path.join(out_loc, "siglip_shot_similarity.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
