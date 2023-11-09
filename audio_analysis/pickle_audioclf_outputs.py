import os
import sys
import pickle
import json
import argparse
import tempfile
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from moviepy.editor import VideoFileClip
sys.path.append("unilm/beats/")
from BEATs import BEATs, BEATsConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Runs audio classification on shot segments")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def read_video_paths(file_path, base_input_dir, base_output_dir):
    ## Input base dir and output base dir are appended with video name such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_models(device, script_dir):
    checkpoint = torch.load(script_dir+"/pretrained_utils/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.to(device)
    BEATs_model.eval()

    with open(script_dir+"/pretrained_utils/ontology.json", "r") as f:
        data = json.load(f)

    idx_to_code = {v: k for k, v in checkpoint['label_dict'].items()}

    label_map = {}
    for entry in data:
        if entry["id"] in idx_to_code:
            label_map[idx_to_code[entry["id"]]] = entry["name"]

    return BEATs_model, label_map


def chop_audio_segments(waveform, sampling_rate, window):
    samples_per_segment = window * sampling_rate  # 10 seconds * 16000 samples/second
    num_full_segments = waveform.shape[0] // samples_per_segment
    segments = []
    ## Full segments
    for i in range(num_full_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = waveform[start:end]
        segments.append(segment.numpy())

    # Add the remaining segment if it exists
    if waveform.shape[0] % samples_per_segment != 0:
        start = num_full_segments * samples_per_segment
        end = waveform.shape[0] 
        segment = waveform[start:end]
        num_zeroes = samples_per_segment - (end - start)
        padded_segment = F.pad(segment, (0, num_zeroes))
        segments.append(padded_segment.numpy())

    return segments


def classify_shot_segments(script_dir, video_path, shots, beats_model, 
                           label_map, device):
    """
    Takes shot segments and perform audio classification into 527 AudioSet Classes
    Classes: https://research.google.com/audioset/dataset/index.html
    Example classes: music, speech, silence, vehicle, animal, etc

    Args:
        script_dir (str): Full path of this script
        video_path (str): Full path to the video file
        shots (list of dicts): List of shot boundaries (Start and end times)
        beats_model (BEATs): BEATs model for audio classification
        label_map (dict): Mapping of label index to label name
        device (torch.device): Device to run the model on

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): shot wise audio classification results [Order of output same as shots]
    """

    video_feat_dict = {"github_repo": "https://github.com/microsoft/unilm",
                        "commit_id": "13641268b59df5cf90d27b451d87ab58b6a07055",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {}
                      }
    
    video = VideoFileClip(video_path+".mp4")
    ## The model is trained using 16kHz audio
    required_sr = 16000

    shot_predictions = []
    for shot in shots:
        start_time = shot["start"]
        end_time = shot["end"]
        if (start_time - end_time) == 0:
            continue

        audio_data = video.audio.subclip(start_time, end_time)
        with tempfile.NamedTemporaryFile(dir=os.path.join(script_dir, "temp/"), suffix=".wav", delete=True) as tmp:
            audio_data.write_audiofile(tmp.name, fps=required_sr)
            waveform, _ = torchaudio.load(tmp.name)
            if waveform.shape[0] > 1:  # if stereo
                waveform = waveform.mean(dim=0)  # convert to mono

            ## Chop audio into further segments of 10 seconds
            audio_segments = torch.tensor(np.array(chop_audio_segments(waveform, required_sr, 10))) # Chop audio into 10s segments
            padding_mask = torch.zeros(len(audio_segments), len(audio_segments[0])).bool()

            audio_segments = audio_segments.to(device)
            padding_mask = padding_mask.to(device)

            with torch.no_grad():
                probs = beats_model.extract_features(audio_segments, padding_mask=padding_mask)[0]
            
            for i, (top3_label_prob, top3_label_idx) in enumerate(zip(*probs.topk(k=3))):
                top3_label = [label_map[label_idx.item()] for label_idx in top3_label_idx]

                shot_predictions.append({"start": start_time, "end": end_time, "top3_label": top3_label, "top3_label_prob": top3_label_prob.tolist()})
        
    video_feat_dict["output_data"] = shot_predictions

    return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    ## Model trained with audio sampled at 16kHz
    beats_model, label_map = get_models(device, script_dir)

    if not os.path.exists(os.path.join(script_dir, "temp/")):
        os.makedirs(os.path.join(script_dir, "temp/"))

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video [{i+1}/{len(input_paths)}]\t{input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "shot_audio_classification_beats.pkl")):
            shot_dict = pickle.load(open(os.path.join(out_loc, "transnet_shotdetection.pkl"), "rb"))

            video_feat_dict = classify_shot_segments(script_dir, input_path, shot_dict["output_data"]["shots"], 
                                                        beats_model, label_map, device)

            with open(os.path.join(out_loc, "shot_audio_classification_beats.pkl"), "wb") as f:
                pickle.dump(video_feat_dict, f)


if __name__ == "__main__":
    main()
