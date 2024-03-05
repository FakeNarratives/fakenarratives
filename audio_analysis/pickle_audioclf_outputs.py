import os
import sys
import pickle
import json
import argparse
import tempfile
import torchaudio
from moviepy.editor import VideoFileClip
from audio_utils import *
sys.path.append("unilm/beats/")
from BEATs import BEATs, BEATsConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Runs audio classification on shot segments")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    parser.add_argument("-r", "--rewrite", action="store_true", help="Rewrite existing files")
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


def aggregate_probs(top3_label_probs):
    """
    Aggregates probabilities of each label across segments
    """
    ## Accumulate probabilities for each label
    label_probs = {}
    for labels, probs in top3_label_probs:
        for l, p in zip(labels, probs):
            if l in label_probs:
                label_probs[l] += p
            else:
                label_probs[l] = p
    
    ## Normalize probabilities
    total_prob = sum(label_probs.values())
    for l in label_probs:
        label_probs[l] /= total_prob
        
    ## Get top 3 labels
    top3_label_probs = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    return top3_label_probs



def classify_shot_segments(script_dir, video_path, video, shots, 
                           beats_model, label_map, device):
    """
    Takes shot segments and perform audio classification into 527 AudioSet Classes
    Classes: https://research.google.com/audioset/dataset/index.html
    Example classes: music, speech, silence, vehicle, animal, etc

    Args:
        script_dir (str): Full path of this script
        video_path (str): Full path of the video
        video (VideoFileClip): VideoFileClip object of the video
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
    
    ## The model is trained using 16kHz audio
    required_sr = 16000

    shot_predictions = []
    for shot in shots:
        start_time, end_time = shot["start"], shot["end"]
        if end_time <= start_time:
            
            shot_predictions.append({"start": start_time, "end": end_time, "top3_label": "None", "top3_label_prob": "None"})
            
            continue

        audio_data = video.audio.subclip(start_time, end_time)
        with tempfile.NamedTemporaryFile(dir=os.path.join(script_dir, "temp/"), suffix=".wav", delete=True) as tmp:
            audio_data.write_audiofile(tmp.name, fps=required_sr, verbose=False, logger=None)
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
            
            top3_label_probs = []
            for i, (top3_label_prob, top3_label_idx) in enumerate(zip(*probs.topk(k=3))):
                top3_label_probs.append(([label_map[label_idx.item()] for label_idx in top3_label_idx], top3_label_prob.tolist()))        
                
            if len(top3_label_probs) > 1:
                top3_label_probs = aggregate_probs(top3_label_probs)
                shot_predictions.append({"start": start_time, "end": end_time, "top3_label": [l for l, _ in top3_label_probs], "top3_label_prob": [p for _, p in top3_label_probs]})
            else:
                shot_predictions.append({"start": start_time, "end": end_time, "top3_label": top3_label_probs[0][0], "top3_label_prob": top3_label_probs[0][1]})
        
    video_feat_dict["output_data"] = shot_predictions

    return video_feat_dict


def classify_speaker_segments(script_dir, video_path, video, speaker_segments, 
                              beats_model, label_map, device):
    """
    Takes speaker segments and perform audio classification into 527 AudioSet Classes
    Classes: https://research.google.com/audioset/dataset/index.html
    Example classes: music, speech, silence, vehicle, animal, etc

    Args:
        script_dir (str): Full path of this script
        video_path (str): Full path of the video
        video (VideoFileClip): VideoFileClip object of the video
        speaker_segments (list of dicts): List of speaker segments (Start and end times)
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
    
    
    ## The model is trained using 16kHz audio
    required_sr = 16000
    
    ## Merge speaker segments with the same speaker ID, called speaker turns
    speaker_segments = get_speaker_turns(speaker_segments)

    segment_predictions = []
    for segment in speaker_segments:
        start_time = segment["start"]
        end_time = segment["end"]
        if end_time <= start_time:
            
            segment_predictions.append({"start": start_time, "end": end_time, "top3_label": "None", "top3_label_prob": "None"})
            continue

        audio_data = video.audio.subclip(start_time, end_time)
        with tempfile.NamedTemporaryFile(dir=os.path.join(script_dir, "temp/"), suffix=".wav", delete=True) as tmp:
            audio_data.write_audiofile(tmp.name, fps=required_sr, verbose=False, logger=None)
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
            
            top3_label_probs = []
            for i, (top3_label_prob, top3_label_idx) in enumerate(zip(*probs.topk(k=3))):
                top3_label_probs.append(([label_map[label_idx.item()] for label_idx in top3_label_idx], top3_label_prob.tolist()))        
                
            if len(top3_label_probs) > 1:
                top3_label_probs = aggregate_probs(top3_label_probs)
                segment_predictions.append({"start": start_time, "end": end_time, "top3_label": [l for l, _ in top3_label_probs], "top3_label_prob": [p for _, p in top3_label_probs]})
            else:
                segment_predictions.append({"start": start_time, "end": end_time, "top3_label": top3_label_probs[0][0], "top3_label_prob": top3_label_probs[0][1]})
        
    video_feat_dict["output_data"] = segment_predictions

    return video_feat_dict, speaker_segments


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seeds(42)

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

        if not args.rewrite:
            print("Already processed. Skipping...")
            continue

        shot_segments = pickle.load(open(os.path.join(out_loc, "transnet_shotdetection.pkl"), "rb"))["output_data"]["shots"]
        whisper_spk_segments = pickle.load(open(os.path.join(out_loc, "asr_whisper.pkl"), "rb"))["output_data"]["speaker_segments"]
        whisperx_spk_segments = pickle.load(open(os.path.join(out_loc, "asr_whisperx.pkl"), "rb"))["output_data"]["speaker_segments"]

        video = VideoFileClip(input_path+".mp4")

        video_feat_dict = classify_shot_segments(script_dir, input_path, video, shot_segments, 
                                                    beats_model, label_map, device)
        
        print("Number of features:", len(video_feat_dict["output_data"]), ", Number of shot segments:", len(shot_segments))
        assert len(video_feat_dict["output_data"]) == len(shot_segments)
        
        video_feat_dict2, speaker_segments2 = classify_speaker_segments(script_dir, input_path, video, whisper_spk_segments, 
                                                    beats_model, label_map, device)
        
        print("Number of features:", len(video_feat_dict2["output_data"]), ", Number of speaker turns:", len(speaker_segments2))
        assert len(video_feat_dict2["output_data"]) == len(speaker_segments2)
        
        video_feat_dict3, speaker_segments3 = classify_speaker_segments(script_dir, input_path, video, whisperx_spk_segments,
                                                    beats_model, label_map, device)
        
        print("Number of features:", len(video_feat_dict3["output_data"]), ", Number of speaker turns:", len(speaker_segments3))
        assert len(video_feat_dict3["output_data"]) == len(speaker_segments3)

        print("Saving to", os.path.join(out_loc, "shot_audioClf.pkl"))
        with open(os.path.join(out_loc, "shot_audioClf.pkl"), "wb") as f:
            pickle.dump(video_feat_dict, f)

        print("Saving to", os.path.join(out_loc, "whisperspeaker_audioClf.pkl"))
        with open(os.path.join(out_loc, "whisperspeaker_audioClf.pkl"), "wb") as f:
            pickle.dump(video_feat_dict2, f)
        
        print("Saving to", os.path.join(out_loc, "whisperxspeaker_audioClf.pkl"))
        with open(os.path.join(out_loc, "whisperxspeaker_audioClf.pkl"), "wb") as f:
            pickle.dump(video_feat_dict3, f)

        print()


if __name__ == "__main__":
    main()
