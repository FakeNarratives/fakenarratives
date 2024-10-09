import os
import sys
import torch
import pickle
import logging
import numpy as np
from PIL import Image
from torch import nn
import decord
from decord import VideoReader, cpu
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoProcessor, AutoModel
from torchvision import models as pt_models
from transformers import AutoImageProcessor, ConvNextV2Model
from torchvision import transforms as trn
from tqdm import tqdm
sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neighboring shots in a video.")
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="number of workers to use for data loading")
    parser.add_argument("-r", "--rewrite", action="store_true", help="rewrite existing pickle files")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=640,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args

def get_models(device, model_types):
    models = {}
    processors = {}
    for model_type in model_types:
        if model_type == "siglip":
            model = AutoModel.from_pretrained("google/siglip-base-patch16-224",
                                              torch_dtype=torch.float16,
                                              device_map=device)
            processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        elif model_type == "convnextv2":
            model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-224")
            processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-base-22k-224")
            model.to(device)
        elif model_type == "places":
            model = pt_models.resnet50(num_classes=365)
            checkpoint = torch.load("places365/resnet50_places365.pth.tar", map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            processor = trn.Compose([
                trn.ToPILImage(),
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            model = nn.Sequential(*list(model.children())[:-1])
            model.to(device)

        models[model_type] = model.eval()
        processors[model_type] = processor
    
    return models, processors

def get_shot_frames(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start_frame"], shots[index]["end_frame"]
    return None, None

def extract_frames(vr, start_frame, end_frame, fps, subsample_rate=4):
    step_size = max(1, int(fps // subsample_rate))
    frames = vr.get_batch(range(start_frame, end_frame + 1))
    frames = frames[::step_size]
    ## Max frames 32 (To avoid GPU OOM Error)
    if len(frames) > 32:  ## Uniformly sample 64 frames
        indices = np.linspace(0, len(frames) - 1, 32, dtype=int)
        frames = [frames[i] for i in indices]
    return frames

def process_frames(frames, model_type, processor, device):
    if model_type == "siglip":
        frames = np.array(frames)
        inputs = processor(images=frames, return_tensors="pt")
        inputs = inputs.to(torch.float16).to(device)
    elif model_type == "convnextv2":
        frames = np.array(frames)
        inputs = processor(frames, return_tensors="pt")
        inputs = inputs.to(device)
    elif model_type == "places":
        tensors = [processor(frame) for frame in frames]
        inputs = torch.stack(tensors)
        inputs = inputs.to(device)
    return inputs

@torch.no_grad()
def get_features(inputs, model_type, model):
    if model_type == "siglip":
        return model.get_image_features(**inputs)
    elif model_type == "convnextv2":
        return model(**inputs).pooler_output
    elif model_type == "places":
        return model(inputs).reshape(inputs.shape[0], 2048)

def compute_similarity(ref_feats, query_inputs, model, model_type):
    query_feats = get_features(query_inputs, model_type, model)
    query_feats = query_feats / query_feats.norm(dim=1)[:, None]
    res = torch.mm(ref_feats, query_feats.T)
    return [res.mean().item(), res.median().item(), res.max().item()]

def process_video(video_path, shot_dict, models, processors, args, device="cuda"):
    """
    Process the video to get shot similarity using all preloaded models.
    Take a image model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        shot_dict (dict): dictionary containing shot information
        models_processors (dict): dictionary containing all loaded models and their processors
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        Multiple video_feat_dicts (dict): dictionary containing shot similarity between two neibhboring shots from the reference shot in a video.
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

    video_feat_dicts = {
        "siglip": {
            "github_repo" : "https://huggingface.co/google/siglip-base-patch16-224",
            "commit_id": "842447fb7dc4db8e8fc3bb0bb09906408eff49ee",
            "parameters": "default",
            "video_file": video_path,
            "output_data": []
        },
        "convnextv2": {
            "github_repo" : "https://huggingface.co/facebook/convnextv2-base-22k-224",
            "commit_id": "4d4e88a3f4d724af643bc78e4e4b66bf2678098a",
            "parameters": "default",
            "video_file": video_path,
            "output_data": []
        },
        "places": {
            "github_repo" : "https://github.com/CSAILVision/places365",
            "commit_id": "06218620d593de09ac4f9f39b72ea0d175990a24",
            "parameters": "default",
            "video_file": video_path,
            "output_data": []
        }
    }
    
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.num_workers, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        # Extract reference frames and process them for all models
        ref_frames = extract_frames(vr, ref_shot["start_frame"], ref_shot["end_frame"], fps, subsample_rate=4)
        ref_inputs = {model_type: process_frames(ref_frames, model_type, processors[model_type], device) for model_type in models.keys()}
        ref_feats = {model_type: get_features(ref_inputs[model_type], model_type, models[model_type]) for model_type in models.keys()}
        ref_feats = {model_type: feats / feats.norm(dim=1)[:, None] for model_type, feats in ref_feats.items()}

        shot_frames = [get_shot_frames(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        # Initialize a similarity dictionary for each model
        similarities = {model_type: {"shot": ref_shot} for model_type in models.keys()}

        for j, (start_frame, end_frame) in enumerate(shot_frames):
            if start_frame is not None and end_frame is not None:
                query_frames = extract_frames(vr, start_frame, end_frame, fps, subsample_rate=4)
                query_inputs = {model_type: process_frames(query_frames, model_type, processors[model_type], device) for model_type in models.keys()}

                for model_type in models.keys():
                    similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                    similarity_scores = compute_similarity(ref_feats[model_type], query_inputs[model_type], models[model_type], model_type)
                    similarities[model_type][similarity_key] = similarity_scores

        # Append results for each model into their respective dictionaries
        for model_type in models.keys():
            video_feat_dicts[model_type]["output_data"].append(similarities[model_type])

    return video_feat_dicts


def main():
    args = parse_args()
    set_seeds(42)
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_types = ["places"]
    models, processors = get_models(device, model_types)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)
        os.makedirs(output_dir, exist_ok=True)

        # Skip if all pickle files already exist
        if all(os.path.exists(os.path.join(output_dir, f"{model_type}_shot_similarity.pkl")) for model_type in model_types) and not args.rewrite:
            logging.info(f"\t[Image Embeddings] Found pkl: {output_dir}, skipping")
            continue

        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")
        if not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot detection file in {shot_detection_path}")
            continue

        with open(shot_detection_path, "rb") as pklfile:
            shot_content = pickle.load(pklfile)

        logging.info(f"\tNumber of shots: {len(shot_content['output_data']['shots'])}")

        video_feat_dicts = process_video(video_path, shot_content, models, processors, args, device)

        # Save each model's output in a separate pickle file
        for model_type in models.keys():
            with open(os.path.join(output_dir, f"{model_type}_shot_similarity.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dicts[model_type], output_file)

        print("%%\n")

if __name__ == "__main__":
    main()
