import os
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
import torch
import pickle
from tqdm import tqdm
import decord
from decord import VideoReader, cpu
from transformers import AutoImageProcessor, VideoMAEModel
from transformers import XCLIPProcessor, XCLIPModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neibhboring shots in a video.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    parser.add_argument("-m", "--model", type=str, default="kinetics-vmae", help="kinetics-vmae | ssv2-vmae | kinetics-xclip")
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


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:# print(image_features.shape, text_features.shape)
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_shot_time(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start"], shots[index]["end"]
    return None, None


def sample_frame_indices(clip_len, indices):
    if len(indices) <= clip_len:
        return indices + [indices[-1]] * (clip_len - len(indices))
    else:
        step_size = max(1, len(indices) // clip_len)
        return [indices[i] for i in range(0, len(indices), step_size)][:clip_len]


def get_subclip_indices(start_time, end_time, fps, total_frames):
    start_index = int(round(start_time * fps))
    end_index = int(round(end_time * fps))
    return [i for i in range(start_index, min(end_index + 1, total_frames))]


def process_frames(frames, processor, model_type):
    if 'vmae' in model_type:
        inputs = processor(list(frames), return_tensors="pt")
    else:
        inputs = processor(videos=list(frames), return_tensors="pt")

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


def process_video(video_path, output_path, model_type, model, processor, device="cuda"):
    """
    Take VideoMAE model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        model_type (str): kinetics-vmae | ssv2-vmae | kinetics-xclip
        model (AutoModel): VideoMAE model
        processor (AutoProcessor): VideoMAE processor
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
                    "prev_1": 0.0,  ## Video level similarity score over 16 frames
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


    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))
    print("Number of shots: ", len(shot_dict["output_data"]["shots"]))

    # Load video
    vr = VideoReader(video_path+".mp4", num_threads=12, ctx=cpu(0), width=480, height=270)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    clip_len = 16 if "vmae" in model_type else 8

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        ## For each shot and reference shot - cls tag similarity from last hidden state
        similarities = {"shot": ref_shot, "prev_1": 0, "prev_2": 0, "next_1": 0, "next_2": 0}

        ## Compute reference shot feats
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        ref_indices = get_subclip_indices(sr_time, er_time, fps, total_frames)
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
        shot_times = [get_shot_time(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        # Extract and compute for each neighboring shot
        for j, (start_time, end_time) in enumerate(shot_times):
            if start_time is not None and end_time is not None:
                query_indices = get_subclip_indices(start_time, end_time, fps, total_frames)
                ## Sample 16 frames from the reference shot
                query_indices = sample_frame_indices(clip_len, query_indices)
                query_frames = vr.get_batch(query_indices)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feat, list(query_frames), model_type, model, processor, device)


        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    model, processor = get_model(args.model, device)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video {i+1}/{len(input_paths)}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "%s_action_shot_similarity.pkl"%(args.model))):
            
            video_feat_dict = process_video(input_path, out_loc, args.model, model, processor, device)

            with open(os.path.join(out_loc, "%s_action_shot_similarity.pkl"%(args.model)), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
