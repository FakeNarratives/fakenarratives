import sys
import torch
import pickle
import logging
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from transformers import WhisperProcessor, WhisperModel
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
sys.path.append("unilm/beats/")
from BEATs import BEATs, BEATsConfig
sys.path.append(".")
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Generates audio similarity between two neighboring shots in a video.")
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
        "-m",
        "--model",
        type=str,
        default="wav2vec2",
        choices=["wav2vec2", "beats", "whisper"],
        help="Model to use for audio feature extraction"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    parser.add_argument(
        "-n",
        "--num_neighbors",
        type=int,
        default=2,
        help="Number of previous/next neighboring shots to compute similarity for, on each side"
    )
    return parser.parse_args()

def get_model(model_type: str, device: str) -> Tuple[Any, Any]:
    if model_type == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    elif model_type == "beats":
        checkpoint = torch.load("analysis_audio/pretrained_utils/BEATs_iter3_plus_AS2M.pt")
        cfg = BEATsConfig(checkpoint['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint['model'])
        processor = None
    elif model_type == "whisper":
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        model = WhisperModel.from_pretrained("openai/whisper-large-v3")

    logging.info(f"Model Loaded: {model_type}")
    model.to(device)
    model.eval()
    return model, processor

def get_neighbor_keys(num_neighbors: int) -> List[Tuple[int, str]]:
    offsets = list(range(-num_neighbors, 0)) + list(range(1, num_neighbors + 1))
    keys = [f"prev_{abs(offset)}" if offset < 0 else f"next_{offset}" for offset in offsets]
    return list(zip(offsets, keys))

def get_waveform(audio_path: str, start_time: float, end_time: float) -> torch.Tensor:
    audio_array, _ = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
    audio_array = torch.tensor(audio_array).unsqueeze(0).to(torch.float32)
    
    if audio_array.shape[1] < 16000:
        audio_array = F.pad(audio_array, (0, 16000 - audio_array.shape[1]), "constant", 0)
    
    if audio_array.shape[1] > 1600000:  # 100 seconds max
        audio_array = audio_array[:, :1600000]
        
    return audio_array

@torch.no_grad()
def get_features(model_type: str, model: Any, processor: Any, audio_array: torch.Tensor, device: str) -> np.ndarray:
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

def get_shot_feature(audio_path: Path, shot: Dict[str, float], model_type: str, model: Any, processor: Any, device: str) -> np.ndarray:
    audio_array = get_waveform(str(audio_path), shot["start"], shot["end"])
    return get_features(model_type, model, processor, audio_array, device)

def process_video(video_path: Path, output_dir: Path, shot_dict: Dict[str, List[Dict[str, float]]],
                  model_type: str, model: Any, processor: Any, device: str, num_neighbors: int = 2) -> Dict[str, Any]:
    """
    Computes audio shot similarity between neighboring shots and the reference shot in a video.
    Reference shot: i, Neighboring shots: i-num_neighbors, ..., i-1, i+1, ..., i+num_neighbors

    Args:
        video_path (Path): Path to the video file
        output_dir (Path): Path to the output directory
        shot_dict (Dict[str, List[Dict[str, float]]]): Dictionary containing shot information
        model_type (str): Type of model used for feature extraction
        model (Any): The loaded model for feature extraction
        processor (Any): The loaded processor for the model
        device (str): The device to run computations on
        num_neighbors (int): Number of previous/next neighboring shots to compute similarity for, on each side

    Returns:
        Dict[str, Any]: Dictionary containing audio shot similarity information
        {
            "github_repo": str,
            "commit_id": str,
            "parameters": "default",
            "video_file": str,
            "output_data": List[Dict[str, Any]]
        }
        Where each item in output_data is:
        {
            "shot": Dict[str, float],
            "prev_1": float,
            "prev_2": float,
            ...
            "next_1": float,
            "next_2": float,
            ...
        }
        Number of prev_N/next_N keys is controlled by num_neighbors.
    """
    video_feat_dict = {
        "github_repo": "",
        "commit_id": "",
        "parameters": "default",
        "video_file": str(video_path),
        "output_data": []
    }
    
    if model_type == "wav2vec2":
        video_feat_dict["github_repo"] = "https://huggingface.co/facebook/wav2vec2-large-xlsr-53"
        video_feat_dict["commit_id"] = "c3f9d884181a224a6ac87bf8885c84d1cff3384f"
    elif model_type == "beats":
        video_feat_dict["github_repo"] = "https://github.com/microsoft/unilm/tree/master/beats"
        video_feat_dict["commit_id"] = "732d834db70ee0fc3886b4bcbcfb4ce7fb829be2"
    elif model_type == "whisper":
        video_feat_dict["github_repo"] = "https://github.com/openai/whisper"
        video_feat_dict["commit_id"] = "ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab"

    audio_path = output_dir / "audio.wav"
    shots = shot_dict["shots"]

    # Extract and cache each shot's feature vector once, instead of recomputing it
    # every time the shot is used as another shot's neighbor.
    shot_feats = [get_shot_feature(audio_path, shot, model_type, model, processor, device)
                  for shot in tqdm(shots, desc="Extracting shot features")]

    neighbor_offsets_keys = get_neighbor_keys(num_neighbors)

    for i, ref_shot in enumerate(tqdm(shots, desc="Computing shot similarities")):
        similarities = {"shot": ref_shot}
        similarities.update({key: 0 for _, key in neighbor_offsets_keys})

        for offset, key in neighbor_offsets_keys:
            j = i + offset
            if 0 <= j < len(shots):
                similarities[key] = cosine_similarity(shot_feats[i], shot_feats[j])[0][0]

        video_feat_dict["output_data"].append(similarities)

    return video_feat_dict

def process_videos(videos: List[str], pkl_dir: str, model_type: str, model: Any, processor: Any, device: str, rewrite: bool, num_neighbors: int = 2) -> Tuple[int, int, List[str]]:
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
        shot_detection_path = output_dir / "transnet_shotdetection.pkl"
        
        if not shot_detection_path.is_file():
            logging.error(f"Missing shot detection file in {shot_detection_path}")
            failed += 1
            failed_videos.append(str(video_path))
            continue

        sim_output_path = output_dir / f"{model_type}_audio_shot_similarity.pkl"
        if sim_output_path.exists() and not rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            with open(shot_detection_path, "rb") as pklfile:
                shot_content = pickle.load(pklfile)["output_data"]
            
            video_feat_dict = process_video(video_path, output_dir, shot_content,
                                            model_type, model, processor, device, num_neighbors)

            with open(sim_output_path, "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

            logging.info(f"Saved similarities to [{vi+1}/{len(videos)}]: {sim_output_path}")
            
            successful += 1
        except Exception as e:
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")
            failed += 1
            failed_videos.append(str(video_path))

    return successful, failed, failed_videos

def main():
    log_file = setup_logging("audio_shot_similarity")
    
    args = parse_args()
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = get_model(args.model, device)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, args.model, model, processor, device, args.rewrite, args.num_neighbors)

    # Log summary
    total = successful + failed
    logging.info(f"Audio shot similarity processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nAudio shot similarity processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()