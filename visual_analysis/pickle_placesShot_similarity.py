import os
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
import torch
import pickle
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from torchvision import models
from torch import nn
from torchvision import transforms as trn
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from memory_profiler import profile

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neibhboring shots in a video.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def get_model(device):
    ## Using Pytorch vision models
    model = models.resnet50(num_classes=365)
    checkpoint = torch.load("places365/resnet50_places365.pth.tar", map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    transforms = trn.Compose([
            trn.Resize((256,256)),
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    feature_extractor.eval()
    feature_extractor.to(device)

    return feature_extractor, transforms


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:# print(image_features.shape, text_features.shape)
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_shot_time(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start"], shots[index]["end"]
    return None, None

def extract_frames(vr, start_time, end_time, fps, subsample_rate=2, processor=None):
    step_size = max(1, int(fps // subsample_rate))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frames = vr.get_batch(range(start_frame, end_frame + 1))
    frames = frames[::step_size]
    tensors = [processor(Image.fromarray(frame.numpy())) for frame in frames]
    return torch.stack(tensors)


def compute_similarity(ref_feats, query_inputs, model):

    with torch.no_grad():
        query_feats = model(query_inputs).reshape(query_inputs.shape[0], 2048)
        query_feats = query_feats / query_feats.norm(dim=1)[:, None]
        res = torch.mm(ref_feats, query_feats.T)

    return [res.mean().item(), res.median().item(), res.max().item()]


def process_video(video_path, output_path, model, processor, device="cuda"):
    """
    Take ResNet50 places model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        model (AutoModel): ResNet50 places model
        processor (AutoProcessor): ResNet50 places transform
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing shot similarity between two neibhboring shots from the reference shot in a video.
        As:
        {
            "github_repo": "https://github.com/CSAILVision/places365",
            "commit_id": "06218620d593de09ac4f9f39b72ea0d175990a24",
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

    video_feat_dict = {"github_repo": "https://github.com/CSAILVision/places365",
                        "commit_id": "06218620d593de09ac4f9f39b72ea0d175990a24",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": []
                        }


    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))
    print("Number of shots: ", len(shot_dict["output_data"]["shots"]))

    # Load the video
    vr = VideoReader(video_path+".mp4", num_threads=12, ctx=cpu(0), width=480, height=270)
    fps = vr.get_avg_fps()

    for i, ref_shot in enumerate(shot_dict["output_data"]["shots"]):
        ## For each shot and reference shot - have [mean, median and max] similarity scores
        similarities = {"shot": ref_shot, "prev_1": [0, 0, 0], "prev_2": [0, 0, 0], "next_1": [0, 0, 0], "next_2": [0, 0, 0]}

        ## Compute reference shot feats
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        ref_inputs = extract_frames(vr, sr_time, er_time, fps, subsample_rate=2, processor=processor).to(device)
        with torch.no_grad():
            ref_feats = model(ref_inputs).reshape(ref_inputs.shape[0], 2048)      ## CLS tag holds the image representation
            ref_feats = ref_feats / ref_feats.norm(dim=1)[:, None]

        # Time frames for the current and neighboring shots
        shot_times = [get_shot_time(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        # Extract and compute for each neighboring shot
        for j, (start_time, end_time) in enumerate(shot_times):
            if start_time is not None and end_time is not None:
                query_inputs = extract_frames(vr, start_time, end_time, fps, subsample_rate=2, processor=processor).to(device)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feats, query_inputs, model)

        # print(i, similarities)
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
        
        if not os.path.exists(os.path.join(out_loc, "places_shot_similarity.pkl")):
            video_feat_dict = process_video(input_path, out_loc, model, processor, device)

            with open(os.path.join(out_loc, "places_shot_similarity.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
