import os
import torch
import pickle
import logging
import numpy as np
from moviepy.editor import AudioFileClip
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from transformers import WhisperProcessor, WhisperModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from audio_utils import *
import sys
sys.path.append("unilm/beats/")
from BEATs import BEATs, BEATsConfig
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generates shot similarity between two neibhboring shots in a video.")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-m", "--model", type=str, default="wav2vec2", help="wav2vec2 | beats | whisper")
    args = parser.parse_args()
    return args


def get_model(model_type, device):
    ## wav2vec2
    if model_type == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    elif model_type == "beats":
        checkpoint = torch.load("audio_analysis/pretrained_utils/BEATs_iter3_plus_AS2M.pt")
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        processor = None
    elif model_type == "whisper":
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperModel.from_pretrained("openai/whisper-large-v3")

    print("\nModel Loaded: ", model_type, "\n")

    model.to(device)
    model.eval()

    return model, processor


def get_shot_time(index, shots):
    if 0 <= index < len(shots):
        return shots[index]["start"], shots[index]["end"]
    return None, None


def get_waveform(audio_clip):
    if audio_clip.duration == 0.0:
        audio_array = torch.zeros(1, 16000).to(torch.float32)
    else:
        audio_array = torch.tensor(audio_clip.to_soundarray()).to(torch.float32).permute(1, 0) ## n_channels, n_samples
    
    if audio_array.shape[0] > 1:  # if stereo
        audio_array = audio_array.mean(dim=0).unsqueeze(0)
    
    ## Audio clip is at-least 1 second long
    if audio_array.shape[1] < 16000:
        audio_array = F.pad(audio_array, (0, 16000 - audio_array.shape[1]), "constant", 0)
    
    ## Audio clip is max 100 seconds long to avoid memory issues
    if audio_array.shape[1] > 1600000:
        audio_array = audio_array[:, :1600000]
        
    return audio_array


@torch.no_grad()
def get_features(model_type, model, processor, audio_array, device):
    if model_type == "beats":
        audio_array = audio_array.to(device)
        padding_mask = torch.zeros(1, audio_array.shape[1]).bool().to(device)
        return model.extract_features(audio_array, padding_mask=padding_mask)[0].mean(1).cpu().numpy()
    elif model_type == "whisper":
        input_features = processor(audio_array.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features
        decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        outputs = model(input_features.to(device), decoder_input_ids=decoder_input_ids.to(device))
        return outputs.encoder_last_hidden_state.mean(1).cpu().numpy()
    elif model_type == "wav2vec2":
        input_values = processor(audio_array.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values.to(device)
        return model(input_values).extract_features.mean(1).cpu().numpy()


def process_video(video_path, shot_dict, model_type, model, processor, device="cuda"):
    """
    Returns:
        video_feat_dict (dict): dictionary containing audio shot similarity between two neibhboring shots from the reference shot in a video.
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
                    "prev_1": 0.0,  ## Single value similarity score
                    "prev_2": 0.0,
                    "next_1": 0.0,
                    "next_2": 0.0
                }
            ]
        }
    """

    video_feat_dict = {"github_repo": "", "commit_id": "", "parameters": "default", "video_file": video_path, "output_data": []}
    
    if model_type == "wav2vec2":
        video_feat_dict["github_repo"] = "https://huggingface.co/facebook/wav2vec2-large-xlsr-53"
        video_feat_dict["commit_id"] = "c3f9d884181a224a6ac87bf8885c84d1cff3384f"
    elif model_type == "beats":
        video_feat_dict["github_repo"] = "https://github.com/microsoft/unilm/tree/master/beats"
        video_feat_dict["commit_id"] = "732d834db70ee0fc3886b4bcbcfb4ce7fb829be2"
    elif model_type == "whisper":
        video_feat_dict["github_repo"] = "https://github.com/openai/whisper"
        video_feat_dict["commit_id"] = "ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab"

    # Load the audio
    audio = AudioFileClip(video_path, fps=16000)

    for i, ref_shot in enumerate(tqdm(shot_dict["shots"])):  
        ## For each shot and reference shot - have one video level similarity score
        similarities = {"shot": ref_shot, "prev_1": 0, "prev_2": 0, "next_1": 0, "next_2": 0}

        ## Compute reference shot feats
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        
        ## Extract audio for the reference shot
        audio_clip = audio.subclip(sr_time, er_time)
        audio_array = get_waveform(audio_clip)
        ref_feat = get_features(model_type, model, processor, audio_array, device)

        # Time frames for the current and neighboring shots
        shot_times = [get_shot_time(i + offset, shot_dict["shots"]) for offset in [-2, -1, 1, 2]]

        # Extract and compute for each neighboring shot
        for j, (start_time, end_time) in enumerate(shot_times):
            if start_time is not None and end_time is not None:
                query_audio = audio.subclip(start_time, end_time)
                query_array = get_waveform(query_audio)
                query_feat = get_features(model_type, model, processor, query_array, device)
                similarity_key = ["prev_2", "prev_1", "next_1", "next_2"][j]
                similarities[similarity_key] = cosine_similarity(ref_feat, query_feat)[0][0]

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
            shot_content = pickle.load(pklfile)["output_data"]
            
        video_feat_dict = process_video(video_path, shot_content, args.model, model, processor, device)

        with open(os.path.join(output_dir, "%s_audio_shot_similarity.pkl"%(args.model)), "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)

        logging.info(f"\tSaved audio shot similarity in {output_dir}/{args.model}_audio_shot_similarity.pkl")

        print("%%\n")


if __name__ == "__main__":
    main()
