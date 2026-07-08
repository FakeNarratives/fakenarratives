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
    parser.add_argument("--model", type=str, required=True, choices=["siglip", "convnextv2", "places"],
                        help="Choose the model to run")
    parser.add_argument("-n", "--num_neighbors", type=int, default=2,
                        help="number of previous/next neighboring shots to compute similarity for, on each side")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=640,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args

def get_model(device, model_type):
    if model_type == "siglip":
        model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384",
                                          torch_dtype=torch.float16,
                                          device_map=device)
        processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    elif model_type == "convnextv2":
        model = ConvNextV2Model.from_pretrained("facebook/convnextv2-base-22k-384")
        processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-base-22k-384")
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
    
    return model.eval(), processor

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

def get_shot_features(vr, shot, fps, model_type, processor, model, device):
    frames = extract_frames(vr, shot["start_frame"], shot["end_frame"], fps, subsample_rate=4)
    inputs = process_frames(frames, model_type, processor, device)
    feats = get_features(inputs, model_type, model)
    feats = feats / feats.norm(dim=1)[:, None]
    return feats

def similarity_stats(ref_feats, query_feats):
    res = torch.mm(ref_feats, query_feats.T)
    return [res.mean().item(), res.median().item(), res.max().item()]

def get_neighbor_keys(num_neighbors):
    offsets = list(range(-num_neighbors, 0)) + list(range(1, num_neighbors + 1))
    keys = [f"prev_{abs(offset)}" if offset < 0 else f"next_{offset}" for offset in offsets]
    return list(zip(offsets, keys))


def process_video(video_path, shot_dict, model, processor, args, device, model_type):
    """
    Process the video to get shot similarity using pretrained model.
    Take a image model and process the video to get shot similarity between neighboring shots and the reference shot in a video.
    Reference shot: i, Neighboring shots: i-args.num_neighbors, ..., i-1, i+1, ..., i+args.num_neighbors

    Args:
        video_path (str): path to video
        shot_dict (dict): dictionary containing shot information
        model (torch.nn.Module): pretrained model
        processor (transformers.Processor): image processor
        args (argparse.Namespace): arguments
        device (str, optional): device to run inference on. Defaults to "cuda".
        model_type (str): model type

    Returns:
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
                    ...
                    "next_1": [0.0, 0.0, 0.0],
                    "next_2": [0.0, 0.0, 0.0],
                    ...
                }
            ]
        }
        Number of prev_N/next_N keys is controlled by args.num_neighbors.
    """

    video_feat_dict = {
        "github_repo": "https://huggingface.co/google/siglip-so400m-patch14-384" if model_type == "siglip" else 
                        "https://huggingface.co/facebook/convnextv2-base-22k-224" if model_type == "convnextv2" else 
                        "https://github.com/CSAILVision/places365",
        "commit_id": "842447fb7dc4db8e8fc3bb0bb09906408eff49ee" if model_type == "siglip" else 
                      "4d4e88a3f4d724af643bc78e4e4b66bf2678098a" if model_type == "convnextv2" else 
                      "06218620d593de09ac4f9f39b72ea0d175990a24",
        "parameters": "default",
        "video_file": video_path,
        "output_data": []
    }
    
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.num_workers, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    shots = shot_dict["output_data"]["shots"]

    # Extract and cache each shot's normalized features once, instead of recomputing
    # them every time a shot is used as another shot's neighbor.
    shot_feats = [get_shot_features(vr, shot, fps, model_type, processor, model, device)
                  for shot in tqdm(shots, desc="Extracting shot features")]

    neighbor_offsets_keys = get_neighbor_keys(args.num_neighbors)

    for i, ref_shot in enumerate(tqdm(shots, desc="Computing shot similarities")):
        # Initialize a similarity dictionary for each model
        similarities = {"shot": ref_shot}
        similarities.update({key: [0.0, 0.0, 0.0] for _, key in neighbor_offsets_keys})

        for offset, key in neighbor_offsets_keys:
            j = i + offset
            if 0 <= j < len(shots):
                similarities[key] = similarity_stats(shot_feats[i], shot_feats[j])

        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict


def main():
    args = parse_args()
    set_seeds(42)
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_type = args.model  # Get the model from argument
    model, processor = get_model(device, model_type)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)
        os.makedirs(output_dir, exist_ok=True)

        # Skip if pickle file already exists
        if os.path.exists(os.path.join(output_dir, f"{model_type}_shot_similarity.pkl")) and not args.rewrite:
            logging.info(f"\t[Image Embeddings] Found pkl: {output_dir}, skipping")
            continue

        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")
        if not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot detection file in {shot_detection_path}")
            continue

        with open(shot_detection_path, "rb") as pklfile:
            shot_content = pickle.load(pklfile)

        logging.info(f"\tNumber of shots: {len(shot_content['output_data']['shots'])}")

        # Process the video (You need to adapt process_video to accept a single model)
        video_feat_dict = process_video(video_path, shot_content, model, processor, args, device, model_type)

        assert len(video_feat_dict["output_data"]) == len(shot_content["output_data"]["shots"])

        # Save output in a pickle file
        with open(os.path.join(output_dir, f"{model_type}_shot_similarity.pkl"), "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)

        print("%%\n")

if __name__ == "__main__":
    main()
