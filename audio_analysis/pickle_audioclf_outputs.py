import os
import pickle
from moviepy.editor import VideoFileClip
import argparse
from tqdm import tqdm
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torchaudio
import torch.nn.functional as F
import numpy as np


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


def get_models(device):
    feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    model.to(device)

    return feature_extractor, model


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


def classify_shot_segments(script_dir, video_path, shots, 
                            feature_extractor, model, device):
    """
    Takes shot segments and perform audio classification into 527 AudioSet Classes
    Classes: https://research.google.com/audioset/dataset/index.html
    Example classes: music, speech, vehicle, explosion, drum, etc.

    Args:
        script_dir (str): Full path of this script
        video_path (str): Full path to the video file
        shots (list of dicts): List of shot boundaries (Start and end times)
        feature_extractor (AST feature extractor): Extracts features from the audio -> Spectrogram
        model (AST model): AST - Audio Spectrogram Transformer model
        device (torch device) - cuda or cpu

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): shot wise audio classification results [Order of output same as shots]
    """

    video_feat_dict = {"github_repo": "https://github.com/YuanGongND/ast",
                        "commit_id": "31088be8a3f6ef96416145c4b8d43c81f99eba7a",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {}
                      }
    
    video = VideoFileClip(video_path+".mp4")
    ## The model is trained using 16kHz audio
    sr = 16000

    shot_predictions = []
    for shot in shots:
        start_time = shot["start"]
        end_time = shot["end"]
        if (start_time - end_time) == 0:
            continue

        audio_data = video.audio.subclip(start_time, end_time)
        with open(os.path.join(script_dir, "temp", "audio.wav"), "w") as tmp:
            audio_data.write_audiofile(tmp.name, fps=sr)
            waveform, sample_rate = torchaudio.load(tmp.name)
            if waveform.shape[0] > 1:  # if stereo
                waveform = waveform.mean(dim=0)  # convert to mono

            ## Chop audio into further segments of 10 seconds as model expects audio of 10 seconds
            segments = chop_audio_segments(waveform, sr, 10)

            inputs = feature_extractor(segments, sampling_rate=sr, return_tensors="pt").input_values
            inputs = inputs.to(device)

            with torch.no_grad():
                logits = model(inputs).logits
            
            predicted_class_ids = torch.argmax(logits, dim=1)
            predicted_labels = [model.config.id2label[pred_cls_id.item()] for pred_cls_id in predicted_class_ids]

            shot_predictions.append({"start": start_time, "end": end_time, "predictions": predicted_labels})

            if os.path.isfile(tmp.name):
                os.remove(tmp.name)
        
    video_feat_dict["output_data"] = shot_predictions

    return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    ## Model trained with audio sampled at 16kHz
    feature_extractor, model = get_models(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(script_dir, "temp/")):
        os.makedirs(os.path.join(script_dir, "temp/"))

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path) 

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "shot_audio_classification.pkl")):
            shot_dict = pickle.load(open(os.path.join(out_loc, "transnet_shotdetection.pkl"), "rb"))

            video_feat_dict = classify_shot_segments(script_dir, input_path, shot_dict["output_data"]["shots"], 
                                                        feature_extractor, model, device)

            with open(os.path.join(out_loc, "shot_audio_classification.pkl"), "wb") as f:
                pickle.dump(video_feat_dict, f)


if __name__ == "__main__":
    main()
