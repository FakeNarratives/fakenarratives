import os
import sys
import torch
import pickle
import json
import logging
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Tuple
sys.path.append("unilm/beats/")
from BEATs import BEATs, BEATsConfig
sys.path.append(".")
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Runs audio classification on shot segments")
    parser.add_argument(
        "-v",
        "--videos",
        nargs="+",
        type=str,
        required=True,
        help="Path to input videos"
    )
    parser.add_argument(
        "-o",
        "--pkl_dir",
        type=str,
        required=True,
        help="Path to pkl directory"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    return parser.parse_args()

def get_models(device: str, script_dir: Path) -> Tuple[BEATs, Dict[int, str]]:
    checkpoint = torch.load(script_dir / "pretrained_utils/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.to(device)
    BEATs_model.eval()

    with open(script_dir / "pretrained_utils/ontology.json", "r") as f:
        data = json.load(f)

    idx_to_code = {v: k for k, v in checkpoint['label_dict'].items()}

    label_map = {}
    for entry in data:
        if entry["id"] in idx_to_code:
            label_map[idx_to_code[entry["id"]]] = entry["name"]

    return BEATs_model, label_map

def aggregate_probs(top3_label_probs: List[Tuple[List[str], List[float]]]) -> List[Tuple[str, float]]:
    """
    Aggregates probabilities of each label across segments
    """
    label_probs = {}
    for labels, probs in top3_label_probs:
        for l, p in zip(labels, probs):
            if l in label_probs:
                label_probs[l] += p
            else:
                label_probs[l] = p
    
    total_prob = sum(label_probs.values())
    for l in label_probs:
        label_probs[l] /= total_prob
        
    top3_label_probs = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    return top3_label_probs

def classify_segments(audio_path: Path, segments: List[Dict[str, float]], 
                      beats_model: BEATs, label_map: Dict[int, str], device: str) -> List[Dict[str, Any]]:
    """
    Takes Shot or Speaker Turn Segments and classifies them into audio categories

    Args:
        audio_path (Path): Path to audio file
        segments (List[Dict[str, float]]): List of segments with start and end times
        beats_model (BEATs): BEATs model
        label_map (Dict[int, str]): Label map for BEATs model
        device (str): Device to run the model on

    Returns:
        List[Dict[str, Any]]: List of segment predictions
    """
    required_sr = 16000
    segment_predictions = []

    for segment in tqdm(segments, desc="Processing segments"):
        start_time, end_time = segment["start"], segment["end"]
        if end_time <= start_time:
            segment_predictions.append({"start": start_time, "end": end_time, "top3_label": None, "top3_label_prob": None})
            continue

        audio_array, _ = librosa.load(str(audio_path), sr=required_sr, offset=start_time, duration=end_time-start_time)
        audio_array = torch.tensor(audio_array).unsqueeze(0).to(torch.float32)

        if audio_array.shape[1] < required_sr:
            audio_array = torch.nn.functional.pad(audio_array, (0, required_sr - audio_array.shape[1]))

        ## Chop audio into further segments of 10 seconds if audio is longer than 10 seconds
        if audio_array.shape[1] > required_sr * 10:   ## (1, N*required_sr*10) --> (N, required_sr*10)
            ceiling_len = (audio_array.shape[1] // (required_sr*10)) * (required_sr*10)
            audio_segments = torch.tensor(np.array([audio_array[:, i:i+required_sr*10] for i in range(0, ceiling_len, required_sr*10)])).squeeze(1)
        else:
            audio_segments = audio_array

        audio_segments = audio_segments.to(device)

        padding_mask = torch.zeros(audio_segments.shape[0], audio_segments.shape[1]).bool().to(device)

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

    return segment_predictions

def process_video(video_path: Path, output_dir: Path, beats_model: BEATs, label_map: Dict[int, str], device: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    video_feat_dict = {
        "github_repo": "https://github.com/microsoft/unilm",
        "commit_id": "13641268b59df5cf90d27b451d87ab58b6a07055",
        "parameters": "default",
        "video_file": str(video_path),
        "output_data": {}
    }

    shot_detection_path = output_dir / "transnet_shotdetection.pkl"
    whisperx_spk_segments_path = output_dir / "asr_whisperx.pkl"
    audio_path = output_dir / "audio.wav"

    if not shot_detection_path.is_file():
        raise FileNotFoundError(f"Missing shot detection file in {shot_detection_path}")

    with open(shot_detection_path, "rb") as pklfile:
        shot_content = pickle.load(pklfile)["output_data"]

    if not whisperx_spk_segments_path.is_file():
        raise FileNotFoundError(f"Missing speaker segments file in {whisperx_spk_segments_path}")
    
    with open(whisperx_spk_segments_path, "rb") as pklfile:
        whisperx_spk_turns = pickle.load(pklfile)["output_data"]["speaker_turns"]

    logging.info(f"Number of shots: {len(shot_content['shots'])}")
    logging.info(f"Number of speaker turns: {len(whisperx_spk_turns)}")

    video_feat_dict["output_data"] = classify_segments(audio_path, shot_content["shots"], beats_model, label_map, device)
    
    video_feat_dict2 = video_feat_dict.copy()
    video_feat_dict2["output_data"] = classify_segments(audio_path, whisperx_spk_turns, beats_model, label_map, device)

    assert len(video_feat_dict["output_data"]) == len(shot_content["shots"])
    assert len(video_feat_dict2["output_data"]) == len(whisperx_spk_turns)

    return video_feat_dict, video_feat_dict2


def process_videos(videos: List[str], pkl_dir: str, beats_model: BEATs, label_map: Dict[int, str], device: str, rewrite: bool) -> Tuple[int, int, List[str]]:
    successful = 0
    failed = 0
    failed_videos = []

    for vi, video in enumerate(videos):
        video_path = Path(video)
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            failed += 1
            failed_videos.append(str(video_path))
            continue
        
        output_dir = Path(pkl_dir) / video_path.stem
        shot_clf_path = output_dir / "shot_audioClf.pkl"
        speaker_clf_path = output_dir / "whisperxspeaker_audioClf.pkl"
        
        if shot_clf_path.exists() and speaker_clf_path.exists() and not rewrite:
            logging.info(f"Output files already exist for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            video_feat_dict, video_feat_dict2 = process_video(video_path, output_dir, beats_model, label_map, device)

            with open(shot_clf_path, "wb") as f:
                pickle.dump(video_feat_dict, f)

            logging.info(f"Saved shot audio classification to [{vi+1}/{len(videos)}] at: {shot_clf_path}")
            
            with open(speaker_clf_path, "wb") as f:
                pickle.dump(video_feat_dict2, f)

            logging.info(f"Saved speaker audio classification to [{vi+1}/{len(videos)}] at: {speaker_clf_path}")
            
            successful += 1
        except Exception as e:
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")
            failed += 1
            failed_videos.append(str(video_path))

    return successful, failed, failed_videos

def main():
    script_path = Path(__file__).resolve()
    log_file = setup_logging("audio_classification")
    
    args = parse_args()
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beats_model, label_map = get_models(device, script_path.parent)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, beats_model, label_map, device, args.rewrite)

    # Log summary
    total = successful + failed
    logging.info(f"Audio classification processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nAudio classification processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()