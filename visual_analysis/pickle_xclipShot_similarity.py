import os
import torch
import pickle
import av
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from transformers import XCLIPProcessor, XCLIPModel
import torch.nn.functional as F
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
    ## Using Microsoft's X-CLIP model
    model = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14-16-frames")
    processor = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14-16-frames")
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


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, indices):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        indices (`List[int]`): List of frame indices to sample from.
    Returns:
        sub_indices (`List[int]`): List of sampled frame indices
    '''
    ## If the number of frames is less than the clip length, repeat last frame to get 16 frames
    if len(indices) <= clip_len:
        return indices + [indices[-1]] * (clip_len - len(indices))
    else:
        step_size = max(1, int(len(indices) // clip_len))
        return [indices[i] for i in range(0, len(indices), step_size)][:clip_len]


def get_subclip_indices(start_time, end_time, fps, video_indices):
    start_index = int(start_time * fps)
    end_index = int(end_time * fps)
    
    return [x for x in video_indices if start_index <= x <= end_index]


def resize_frames(frames, size=(336, 336)):
    resized_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        resized_img = img.resize(size)
        resized_frames.append(np.array(resized_img))
    return resized_frames


def compute_similarity(ref_feats, query_video, model, processor, device):
    query_inputs = processor(videos=resize_frames(query_video), return_tensors="pt", do_reisze=False, do_center_crop=False).to(device)
    
    with torch.no_grad():
        query_feats = model.get_video_features(**query_inputs).cpu().numpy()
        similarity = cosine_similarity(ref_feats, query_feats)

    return similarity[0][0]


def process_video(video_path, output_path, model, processor, device="cuda"):
    """
    Take XCLIP model and process the video to get shot similarity between two neibhboring shots from the reference shot in a video.
    Reference shot: i, Neighboring shots: i-2, i+1, i+1, i+2

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        model (XCLIPModel): XCLIP model
        processor (XCLIPProcessor): XCLIP processor
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing shot similarity between two neibhboring shots from the reference shot in a video.
        As:
        {
            "github_repo": "https://huggingface.co/microsoft/xclip-large-patch14-16-frames",
            "commit_id": "e818169d0ec9bdbab85201110e601c5a588321fa",
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

    video_feat_dict = {"github_repo": "https://huggingface.co/microsoft/xclip-large-patch14-16-frames",
                        "commit_id": "e818169d0ec9bdbab85201110e601c5a588321fa",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": []
                        }


    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))

    # Load the video
    # vr = VideoReader(video_path+".mp4", num_threads=4, ctx=cpu(0))
    container = av.open(video_path+".mp4")
    fps = container.streams.video[0].average_rate

    for i, ref_shot in enumerate(shot_dict["output_data"]["shots"]):
        ## For each shot and reference shot - have one video level similarity score
        similarities = {"shot": ref_shot, "prev_1": 0, "prev_2": 0, "next_1": 0, "next_2": 0}

        ## Compute reference shot feats
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        ref_indices = get_subclip_indices(sr_time, er_time, fps, range(container.streams.video[0].frames))
        ## Sample 16 frames from the reference shot
        ref_indices = sample_frame_indices(16, ref_indices)
        ref_video = read_video_pyav(container, ref_indices)
        print(ref_video.shape)
        ref_inputs = processor(videos=resize_frames(ref_video), return_tensors="pt", do_reisze=False, do_center_crop=False).to(device)
        with torch.no_grad():
            ref_feats = model.get_video_features(**ref_inputs).cpu().numpy()

        # Time frames for the current and neighboring shots
        shot_times = [get_shot_time(i + offset, shot_dict["output_data"]["shots"]) for offset in [-2, -1, 1, 2]]

        # Extract and compute for each neighboring shot
        for j, (start_time, end_time) in enumerate(shot_times):
            if start_time is not None and end_time is not None:
                query_indices = get_subclip_indices(start_time, end_time, fps, range(container.streams.video[0].frames))
                ## Sample 16 frames from the reference shot
                query_indices = sample_frame_indices(16, query_indices)
                query_video = read_video_pyav(container, query_indices)
                print(query_video.shape)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = compute_similarity(ref_feats, query_video, model, processor, device)

        print(similarities)
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

        if not os.path.exists(os.path.join(out_loc, "xclip_shot_similarity.pkl")):
            
            video_feat_dict = process_video(input_path, out_loc, model, processor, device)

            with open(os.path.join(out_loc, "xclip_shot_similarity.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
