import os
import pickle
import torch
import argparse
import yaml
import logging
import whisperx
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

sys.path.append(".")
from audio_utils import get_speaker_turns
from project_utils import set_seeds, setup_logging

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs WhisperX on news videos for automatic speech recognition and speaker diarization")
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-r", "--rewrite", action="store_true", help="Rewrite existing files")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size for whisperX")
    return parser.parse_args()

def get_model(device: str, config: Dict[str, Any]) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    model = whisperx.load_model("large-v3", device=device, compute_type="float16", language="de")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=config['huggingface']['token'], device=device)
    alignment_model, metadata = whisperx.load_align_model(language_code="de", device=device)
    return model, diarize_model, alignment_model, metadata

def transcribe_video(video_path: str, output_dir: str, model: Any, diarize_model: Any, 
                     alignment_model: Any, metadata: Dict[str, Any], device: str, 
                     batch_size: int) -> Dict[str, Any]:
    """
    Transcribes video using WhisperX library and performs speaker diarization.

    Args:
        video_path (str): Full path to the video file
        output_dir (str): Path to the output directory
        model (Any): Initialized Whisperx ASR model
        diarize_model (Any): Initialized Whisperx diarization model
        alignment_model (Any): Initialized Whisperx alignment model
        metadata (Dict[str, Any]): Metadata of the alignment model
        device (str): Device to run the model on

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
        "parameters": "default",
        "video_file": video_path,
    }

    audio_path = os.path.join(output_dir, "audio.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=batch_size, language="de")  ## Transcribe the audio
    result = whisperx.align(result["segments"], alignment_model, metadata, audio, device, return_char_alignments=False)  ## Align the ASR output
    aligned_segments = result["segments"]

    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result) ## Assign speakers to segments/words

    speaker_turns = get_speaker_turns(result["segments"])  ## Get speaker turns from the segments --> Merge consecutive segments with same speaker

    asr_result = {
        "language": "de",
        "text": " ".join(seg["text"].strip() for seg in aligned_segments),
        "segments": aligned_segments,  # List of dict_keys(['start', 'end', 'text', 'words'])
        "speaker_segments": result["segments"],  # List of dict_keys(['start', 'end', 'text', 'words', 'speaker'])
        "word_segments": result["word_segments"],  # List of dict_keys(['word', 'start', 'end', 'score', 'speaker'])
        "speaker_turns": speaker_turns  # List of dict_keys(['start', 'end', 'text', 'speaker', 'n_words'])
    }

    video_feat_dict["output_data"] = asr_result

    return video_feat_dict

def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    log_file = setup_logging("asr_output")

    set_seeds(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = script_path.parent / "config.yml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model, diarize_model, alignment_model, metadata = get_model(device, config)

    for vi, video_path in enumerate(args.videos):
        logging.info(f"Processing video [{vi+1}/{len(args.videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        asr_output_path = os.path.join(output_dir, "asr_whisperx.pkl")
        if os.path.exists(asr_output_path) and not args.rewrite:
            logging.info(f"Output file already exists. Skipping...")
            continue

        os.makedirs(output_dir, exist_ok=True)

        try:
            video_feat_dict = transcribe_video(video_path, output_dir, model, diarize_model, 
                                               alignment_model, metadata, device, args.batch_size)

            logging.info(f"Saving to {asr_output_path}")
            with open(asr_output_path, "wb") as f:
                pickle.dump(video_feat_dict, f)
        except Exception as e:
            logging.error(f"Error processing video {video_path}: {str(e)}")

    logging.info("ASR processing complete.")
    print(f"\nLog file saved at: {log_file}")

if __name__ == "__main__":
    main()