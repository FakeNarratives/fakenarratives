import os
import pickle
import torch
import argparse
import yaml
import logging
import whisperx
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

sys.path.append(".")
from audio_utils import get_speaker_turns
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Runs WhisperX on news videos for automatic speech recognition and speaker diarization")
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
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for whisperX"
    )
    return parser.parse_args()


def get_model(device: str, config: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    model = whisperx.load_model("large-v3", device=device, compute_type="float16")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=config['huggingface']['token'], device=device)
    return model, diarize_model

def transcribe_video(video_path: Path, output_dir: Path, model: Any, diarize_model: Any, 
                        device: str, batch_size: int) -> Dict[str, Any]:
    """
    Transcribes video using WhisperX library and performs speaker diarization.

    Args:
        video_path (Path): Full path to the video file
        output_dir (Path): Path to the output directory
        model (Any): Initialized Whisperx ASR model
        diarize_model (Any): Initialized Whisperx diarization model
        device (str): Device to run the model on
        batch_size (int): Batch size for whisperX

    Returns:
        Dict[str, Any]: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (Dict[str, Any]): Transcription and diarization result from the Whisper model as a dictionary with keys: 
                    ['language', 'text', 'segments', 'speaker_segments', 'word_segments', 'speaker_turns']
    """
    video_feat_dict = {
        "github_repo": "https://github.com/m-bain/whisperX",
        "commit_id": "f2da2f858e99e4211fe4f64b5f2938b007827e17",
        "parameters": f"batch_size={batch_size}",
        "video_file": str(video_path),
    }

    audio_path = output_dir / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = whisperx.load_audio(str(audio_path))

    result = model.transcribe(audio, batch_size=batch_size)
    language_code = result["language"]
    alignment_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result = whisperx.align(result["segments"], alignment_model, metadata, audio, device, return_char_alignments=False)
    aligned_segments = result["segments"]

    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    speaker_turns = get_speaker_turns(result["segments"])

    asr_result = {
        "language": language_code,
        "text": " ".join(seg["text"].strip() for seg in aligned_segments),
        "segments": aligned_segments,
        "speaker_segments": result["segments"],
        "word_segments": result["word_segments"],
        "speaker_turns": speaker_turns
    }

    video_feat_dict["output_data"] = asr_result

    return video_feat_dict


def process_videos(videos: List[str], pkl_dir: str, model: Any, diarize_model: Any, 
                   device: str, batch_size: int, rewrite: bool) -> Tuple[int, int, List[str]]:
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
        asr_output_path = output_dir / "asr_whisperx.pkl"
        
        if asr_output_path.exists() and not rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            video_feat_dict = transcribe_video(video_path, output_dir, model, diarize_model, 
                                               device, batch_size)

            with open(asr_output_path, "wb") as f:
                pickle.dump(video_feat_dict, f)

            logging.info(f"Saved ASR output to [{vi+1}/{len(videos)}]: {asr_output_path}")
            
            successful += 1
        except Exception as e:
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")
            failed += 1
            failed_videos.append(str(video_path))

    return successful, failed, failed_videos

def main():
    script_path = Path(__file__).resolve()
    log_file = setup_logging("asr_output")
    
    args = parse_args()
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = script_path.parent / "config.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model, diarize_model = get_model(device, config)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, model, diarize_model, 
                                                        device, args.batch_size, args.rewrite)

    # Log summary
    total = successful + failed
    logging.info(f"ASR processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  - {video}")
    
    # Print summary to console
    print(f"\nASR processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()