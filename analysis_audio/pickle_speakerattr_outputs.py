import os
import sys
import torch
import pickle
import logging
import argparse
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any, Tuple
from speechbrain.pretrained.interfaces import foreign_class
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

sys.path.append(".")
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Runs speech emotion and gender classification on speaker diarization segments")
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

def get_models(device: str) -> Tuple[Any, Any, Any]:
    run_opts = {"device": device}
    emo_model = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", 
                              classname="CustomEncoderWav2vec2Classifier", run_opts=run_opts)

    gen_model = Wav2Vec2ForSequenceClassification.from_pretrained('alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')
    gen_proc = Wav2Vec2FeatureExtractor.from_pretrained('alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')
    gen_model.to(device)
    gen_model.eval()

    return emo_model, gen_model, gen_proc


def classify_segments(audio_path: Path, speaker_turns: List[Dict[str, Any]], 
                      emo_model: Any, gen_model: Any, gen_proc: Any, 
                      emo_lab_map: Dict[str, str], gen_lab_map: Dict[str, str], 
                      device: str) -> List[Dict[str, Any]]:
    """
    Takes Speaker Diarization segments and classifies them into speech emotion and gender categories.

    Args:
        audio_path (Path): Path to the audio file
        speaker_turns (List[Dict[str, Any]]): List of speaker segments from ASR
        emo_model (Any): Emotion recognition model from speechbrain
        gen_model (Any): Gender recognition model from transformers
        gen_proc (Any): Gender recognition processor from transformers
        emo_lab_map (Dict[str, str]): Label map for emotion recognition
        gen_lab_map (Dict[str, str]): Label map for gender recognition
        device (str): Device to run the model on

    Returns:
        List[Dict[str, Any]]: List of segment predictions
    """

    required_sr = 16000
    segment_predictions = []

    for seg in tqdm(speaker_turns, desc="Processing segments"):
        if seg["end"] <= seg["start"]:
            segment_predictions.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker", "None"),
                "gender_pred": None,
                "gender_prob": None,
                "emotion_pred_top3": None,
                "emotion_prob_top3": None
            })
            continue

        audio_array, _ = librosa.load(str(audio_path), sr=required_sr, offset=seg["start"], duration=seg["end"]-seg["start"])
        audio_array = torch.tensor(audio_array).unsqueeze(0).to(torch.float32)

        if audio_array.shape[1] < required_sr:
            audio_array = torch.nn.functional.pad(audio_array, (0, required_sr - audio_array.shape[1]))

        ## Chop audio into further segments of 10 seconds if audio is longer than 10 seconds
        if audio_array.shape[1] > required_sr * 10:
            ceiling_len = (audio_array.shape[1] // (required_sr*10)) * (required_sr*10)
            audio_segments = torch.tensor(np.array([audio_array[:, i:i+required_sr*10] for i in range(0, ceiling_len, required_sr*10)])).squeeze(1)  ## --> (N, 1600000)
        else:
            audio_segments = audio_array

        audio_segments = audio_segments.to(device)

        input_values = gen_proc(audio_segments, sampling_rate=required_sr, return_tensors='pt').input_values.squeeze(0) ## --> (N, 1600000)
        input_values = input_values.to(device)

        with torch.no_grad():
            result = gen_model(input_values).logits.softmax(dim=1)
            sum_res = result.mean(dim=0)
            max_i = sum_res.detach().cpu().argmax().item()
            gen_prob = sum_res.detach().cpu().max().item()
            gen_pred = gen_lab_map[str(max_i)]

            _, emo_prob, _, text_lab = emo_model.classify_batch(audio_segments)

        if len(text_lab) > 1:
            top_probs = emo_prob.detach().cpu().topk(min(len(text_lab), 3)).values.tolist()
            top_preds = [emo_lab_map[text_lab[i]] for i in emo_prob.detach().cpu().topk(min(len(text_lab), 3)).indices.tolist()]
        else:
            top_probs = [emo_prob.item()]
            top_preds = [emo_lab_map[text_lab[0]]]

        segment_predictions.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": seg.get("speaker", "None"),
            "gender_pred": gen_pred,
            "gender_prob": gen_prob,
            "emotion_pred_top3": top_preds,
            "emotion_prob_top3": top_probs
        })

    return segment_predictions


def process_video(video_path: Path, output_dir: Path, emo_model: Any, gen_model: Any, gen_proc: Any, 
                  emo_lab_map: Dict[str, str], gen_lab_map: Dict[str, str], device: str) -> Dict[str, Any]:
    video_feat_dict = {
        "github_repo": "https://github.com/speechbrain/speechbrain;https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
        "commit_id": "481e5dfddd70d81714b1dea32e5dbfdee7c50c03;0a5d4dc65986030a703faf1942d51a9734824882",
        "parameters": "default",
        "video_file": str(video_path),
        "output_data": {}
    }

    audio_path = output_dir / "audio.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    whisperx_spk_segments_path = output_dir / "asr_whisperx.pkl"
    if not whisperx_spk_segments_path.is_file():
        raise FileNotFoundError(f"Missing speaker turns file in {whisperx_spk_segments_path}")
    
    with open(whisperx_spk_segments_path, "rb") as pklfile:
        whisperx_spk_turns = pickle.load(pklfile)["output_data"]["speaker_turns"]

    logging.info(f"Number of speaker turns: {len(whisperx_spk_turns)}")

    video_feat_dict["output_data"] = classify_segments(audio_path, whisperx_spk_turns, 
                                                       emo_model, gen_model, gen_proc, 
                                                       emo_lab_map, gen_lab_map, device)

    assert len(video_feat_dict["output_data"]) == len(whisperx_spk_turns)

    return video_feat_dict

def process_videos(videos: List[str], pkl_dir: str, emo_model: Any, gen_model: Any, 
                   gen_proc: Any, emo_lab_map: Dict[str, str], gen_lab_map: Dict[str, str], 
                   device: str, rewrite: bool) -> Tuple[int, int, List[str]]:
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
        output_file = output_dir / "whisperxspeaker_segmentClf.pkl"
        
        if output_file.exists() and not rewrite:
            logging.info(f"Output file already exists for {video_path}. Skipping.")
            successful += 1
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
            video_feat_dict = process_video(video_path, output_dir, emo_model, gen_model, gen_proc, 
                                            emo_lab_map, gen_lab_map, device)

            with open(output_file, "wb") as f:
                pickle.dump(video_feat_dict, f)

            logging.info(f"Saved speaker attribute classification to [{vi+1}/{len(videos)}]: {output_file}")
            successful += 1
        except Exception as e:
            logging.error(f"Error processing video [{vi+1}/{len(videos)}]: {video_path}: {str(e)}")
            failed += 1
            failed_videos.append(str(video_path))

    return successful, failed, failed_videos

def main():
    args = parse_args()
    script_path = Path(__file__).resolve()
    log_file = setup_logging("speakerattr_classification")
    
    logging.info(f"Log file will be saved at: {log_file}")
    
    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emo_lab_map = {"neu": "Neutral", "hap": "Happy", "ang": "Angry", "sad": "Sad"}
    gen_lab_map = {"0": "Female", "1": "Male"}

    emo_model, gen_model, gen_processor = get_models(device)

    successful, failed, failed_videos = process_videos(args.videos, args.pkl_dir, emo_model, gen_model, 
                                                       gen_processor, emo_lab_map, gen_lab_map, device, args.rewrite)

    # Log summary
    total = successful + failed
    logging.info(f"Segment classification processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    
    if failed > 0:
        logging.info("Failed videos:")
        for video in failed_videos:
            logging.info(f"  {video}")
    
    # Print summary to console
    print(f"\nSegment classification processing summary:")
    print(f"Total videos processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nLog file saved at: {log_file}")
    if failed > 0:
        print(f"Check the log file for details on failed videos.")

if __name__ == "__main__":
    main()