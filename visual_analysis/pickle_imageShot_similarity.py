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
from torchvision import models
from transformers import AutoImageProcessor, ConvNextV2Model
from torchvision import transforms as trn
from tqdm import tqdm
sys.path.append(".")
from video_utils import read_video_and_get_info
from visual_utils import set_seeds
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neighboring shots in a video.")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-m", "--model", type=str, default="siglip", choices=["siglip", "convnextv2", "places"], help="Model type to use")
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
        model = AutoModel.from_pretrained("google/siglip-large-patch16-384", 
                                          attn_implementation="flash_attention_2",
                                          torch_dtype=torch.float16,
                                          device_map=device)
        processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
    elif model_type == "convnextv2":
        model = ConvNextV2Model.from_pretrained("facebook/convnextv2-large-22k-384")
        processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-large-22k-384")
        model.to(device)
    elif model_type == "places":
        model = models.resnet50(num_classes=365)
        checkpoint = torch.load("places365/resnet50_places365.pth.tar", map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        processor = trn.Compose([
            trn.ToPILImage(),
            trn.Resize((440, 440)),
            trn.CenterCrop(384),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model = nn.Sequential(*list(model.children())[:-1])
        model.to(device)

    model.eval()
    
    return model, processor


def get_shot_frames(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start_frame"], shots[index]["end_frame"]
    return None, None


def extract_frames(vr, start_frame, end_frame, fps, subsample_rate=2):
    step_size = max(1, int(fps // subsample_rate))
    frames = vr.get_batch(range(start_frame, end_frame + 1))
    frames = frames[::step_size]
    ## Max frames 64 (To avoid GPU OOM Error)
    if len(frames) > 64: ## Uniformly sample 64 frames
        indices = np.linspace(0, len(frames) - 1, 64, dtype=int)
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


def process_video(video_path, shot_dict, model, processor, args, device="cuda"):
    """
    Take a image model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        shot_dict (dict): dictionary containing shot information
        model (AutoModel): SigLIP model
        processor (AutoProcessor): SigLIP processor
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
                    "prev_1": [0.0, 0.0, 0.0],  ## [mean, median, max] similarity scores
                    "prev_2": [0.0, 0.0, 0.0],
                    "next_1": [0.0, 0.0, 0.0],
                    "next_2": [0.0, 0.0, 0.0]
                }
            ]
        }
    """
    
    video_feat_dict = {"github_repo": "", "commit_id": "", "parameters": "default", "video_file": video_path, "output_data": []}
    model_type = args.model
    
    if model_type == "siglip":
        video_feat_dict["github_repo"] = "https://huggingface.co/google/siglip-large-patch16-384"
        video_feat_dict["commit_id"] = "842447fb7dc4db8e8fc3bb0bb09906408eff49ee"
    elif model_type == "convnextv2":
        video_feat_dict["github_repo"] = "https://huggingface.co/facebook/convnextv2-large-22k-384"
        video_feat_dict["commit_id"] = "4d4e88a3f4d724af643bc78e4e4b66bf2678098a"
    elif model_type == "resnet50_places":
        video_feat_dict["github_repo"] = "https://github.com/CSAILVision/places365"
        video_feat_dict["commit_id"] = "06218620d593de09ac4f9f39b72ea0d175990a24"

    # Load video
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        similarities = {"shot": ref_shot, "prev_1": [0, 0, 0], "prev_2": [0, 0, 0], "next_1": [0, 0, 0], "next_2": [0, 0, 0]}

        ref_frames = extract_frames(vr, ref_shot["start_frame"], ref_shot["end_frame"], fps, subsample_rate=2)
        ref_inputs = process_frames(ref_frames, model_type, processor, device)
        with torch.no_grad():
            ref_feats = get_features(ref_inputs, model_type, model)
            ref_feats = ref_feats / ref_feats.norm(dim=1)[:, None]

        shot_frames = [get_shot_frames(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        for j, (start_frame, end_frame) in enumerate(shot_frames):
            if start_frame is not None and end_frame is not None:
                query_frames = extract_frames(vr, start_frame, end_frame, fps, subsample_rate=2)
                query_inputs = process_frames(query_frames, model_type, processor, device)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feats, query_inputs, model, model_type)

        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict


def main():
    args = parse_args()

    set_seeds(42)

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_model(device, args.model)

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

        video_feat_dict = process_video(video_path, shot_content, model, processor, args, device)

        with open(os.path.join(output_dir, f"{args.model}_shot_similarity.pkl"), "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
