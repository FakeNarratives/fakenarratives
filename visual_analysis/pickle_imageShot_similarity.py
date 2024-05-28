import os
import torch
import pickle
import numpy as np
from PIL import Image
from torch import nn
import decord
from decord import VideoReader, cpu
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModel
from torchvision import models
from transformers import AutoImageProcessor, ConvNextV2Model
from torchvision import transforms as trn
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neighboring shots in a video.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    parser.add_argument("-m", "--model", type=str, default="siglip", choices=["siglip", "convnextv2", "places"], help="Model type to use")
    args = parser.parse_args()
    return args


def get_model(device, model_type):
    if model_type == "siglip":
        model = AutoModel.from_pretrained("google/siglip-large-patch16-384")
        processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
    elif model_type == "convnextv2":
        model = ConvNextV2Model.from_pretrained("facebook/convnextv2-large-22k-384")
        processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-22k-384")
    elif model_type == "places":
        model = models.resnet50(num_classes=365)
        checkpoint = torch.load("places365/resnet50_places365.pth.tar", map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        processor = trn.Compose([
            trn.Resize((440, 440)),
            trn.CenterCrop(384),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model = nn.Sequential(*list(model.children())[:-1])

    model.eval()
    model.to(device)

    return model, processor


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
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
    ## Max frames 64 (To avoid GPU OOM Error)
    if len(frames) > 64: ## Uniformly sample 64 frames
        indices = np.linspace(0, len(frames) - 1, 64, dtype=int)
        frames = [frames[i] for i in indices]
    return frames


def process_frames(frames, model_type, processor):
    if model_type == "siglip":
        inputs = processor(images=frames, return_tensors="pt")
    elif model_type == "convnextv2":
        inputs = processor(list(frames), return_tensors="pt")
    elif model_type == "places":
        tensors = [processor(Image.fromarray(frame.numpy())) for frame in frames]
        inputs = torch.stack(tensors)
    return inputs


def get_features(inputs, model_type, model):
    if model_type == "siglip":
        return model.get_image_features(**inputs)
    elif model_type == "convnextv2":
        return model(**inputs).pooler_output
    elif model_type == "places":
        return model(inputs).reshape(inputs.shape[0], 2048)


def compute_similarity(ref_feats, query_inputs, model, model_type):
    with torch.no_grad():
        query_feats = get_features(query_inputs, model_type, model)
        query_feats = query_feats / query_feats.norm(dim=1)[:, None]
        res = torch.mm(ref_feats, query_feats.T)
    return [res.mean().item(), res.median().item(), res.max().item()]


def process_video(video_path, output_path, model, processor, device="cuda", model_type="siglip"):
    """
    Take a image model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        model (AutoModel): SigLIP model
        processor (AutoProcessor): SigLIP processor
        device (str, optional): device to run inference on. Defaults to "cuda".
        model_type (str, optional): model type. Defaults to "siglip".

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
                    "prev_1": [0.0, 0.0, 0.0],  ## [mean, median, max] similarity scores
                    "prev_2": [0.0, 0.0, 0.0],
                    "next_1": [0.0, 0.0, 0.0],
                    "next_2": [0.0, 0.0, 0.0]
                }
            ]
        }
    """
    
    video_feat_dict = {"github_repo": "", "commit_id": "", "parameters": "default", "video_file": video_path, "output_data": []}
    
    if model_type == "siglip":
        video_feat_dict["github_repo"] = "https://huggingface.co/google/siglip-large-patch16-384"
        video_feat_dict["commit_id"] = "842447fb7dc4db8e8fc3bb0bb09906408eff49ee"
    elif model_type == "convnextv2":
        video_feat_dict["github_repo"] = "https://huggingface.co/facebook/convnextv2-large-22k-384"
        video_feat_dict["commit_id"] = "4d4e88a3f4d724af643bc78e4e4b66bf2678098a"
    elif model_type == "resnet50_places":
        video_feat_dict["github_repo"] = "https://github.com/CSAILVision/places365"
        video_feat_dict["commit_id"] = "06218620d593de09ac4f9f39b72ea0d175990a24"

    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))
    print("Number of shots: ", len(shot_dict["output_data"]["shots"]))

    vr = VideoReader(video_path + ".mp4", num_threads=12, ctx=cpu(0), width=480, height=270)
    fps = vr.get_avg_fps()

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        similarities = {"shot": ref_shot, "prev_1": [0, 0, 0], "prev_2": [0, 0, 0], "next_1": [0, 0, 0], "next_2": [0, 0, 0]}

        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        ref_frames = extract_frames(vr, sr_time, er_time, fps, subsample_rate=2)
        ref_inputs = process_frames(ref_frames, model_type, processor).to(device)
        with torch.no_grad():
            ref_feats = get_features(ref_inputs, model_type, model)
            ref_feats = ref_feats / ref_feats.norm(dim=1)[:, None]

        shot_times = [get_shot_time(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        for j, (start_time, end_time) in enumerate(shot_times):
            if start_time is not None and end_time is not None:
                query_frames = extract_frames(vr, start_time, end_time, fps, subsample_rate=2)
                query_inputs = process_frames(query_frames, model_type, processor).to(device)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feats, query_inputs, model, model_type)

        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    model, processor = get_model(device, args.model)

    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video {i+1}/{len(input_paths)}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        output_file_name = f"{args.model}_shot_similarity.pkl"
        if not os.path.exists(os.path.join(out_loc, output_file_name)):
            video_feat_dict = process_video(input_path, out_loc, model, processor, device, args.model)

            with open(os.path.join(out_loc, output_file_name), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
