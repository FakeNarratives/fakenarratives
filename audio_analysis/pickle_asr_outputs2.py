import os
import pickle
import tempfile
from moviepy.editor import VideoFileClip
import torch
import argparse
import sys
import whisperx
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Runs WhisperX on news videos for automatic speech recognition and speaker diarization")
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


def get_model(device, config):
    model = whisperx.load_model("large-v2", device=device, compute_type="float16")

    diarize_model = whisperx.DiarizationPipeline(use_auth_token=config['huggingface']['token'], device=device)

    alignment_model, metadata = whisperx.load_align_model(language_code="de", device=device)

    return model, diarize_model, alignment_model, metadata


def transcribe_video(script_dir, video_path, model, diarize_model, 
                        alignment_model, metadata, device, config):
    """
    Transcribes video using WhisperX library and performs speaker diarization.

    The dictionary includes:
    1. GitHub repo of the Whisper model
    2. Commit ID of the used Whisper model
    3. Parameters used (default in this case)
    4. Video file path
    5. Transcription result

    Args:
        script_dir (str): Full path of this script
        video_path (str): Full path to the video file
        model (whisperx model): Initialized Whisperx ASR model
        diarize_model (whisperx model): Initialized Whisperx diarization model
        alignment_model (whisperx model): Initialized Whisperx alignment model
        metadata (dict): Metadata of the alignment model
        device (str): Device to run the model on

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (str): Transcription and diarization result from the Whisper model as a dictionary with keys: ['text', 'segments', 'language', 'speaker_segments']
    """

    video_feat_dict = {"github_repo": "https://github.com/m-bain/whisperX",
                        "commit_id": "4ece2369d73465a4e50417d0f0a45e262343703a",
                        "parameters": "default",
                        "video_file": video_path,
                      }

    video = VideoFileClip(video_path+".mp4")

    asr_result = {}

    ## Changed temporary audio files from .mp3 to .wav to avoid loading errors
    with tempfile.NamedTemporaryFile(dir=os.path.join(script_dir, "temp/"), suffix=".wav", delete=True) as tmp:
        video.audio.write_audiofile(tmp.name)
        audio = whisperx.load_audio(tmp.name)

        # result = model.transcribe(tmp.name)
        result = model.transcribe(audio, batch_size=config["whisperx"]["batch_size"], language="de")
        text = ""
        for seg in result["segments"]:
            text += seg["text"].strip() + " "
        asr_result["text"] = text.strip()
        asr_result["segments"] = result["segments"]

        diarize_segments = diarize_model(audio)

        result = whisperx.align(result["segments"], alignment_model, metadata, audio, device, return_char_alignments=False)

        result = whisperx.assign_word_speakers(diarize_segments, result)

    ## Save results as a dictionary: dict_keys(['text', 'segments', 'language', 'speaker_segments'])
    asr_result["speaker_segments"] = result["segments"]
    asr_result["language"] = "de"
    asr_result["word_segments"] = result["word_segments"]

    video_feat_dict["output_data"] = asr_result

    return video_feat_dict


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))

    ## Load config file
    with open(os.path.join(script_dir, "config.yml"), 'r') as file:
        config = yaml.safe_load(file)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    model, diarize_model, alignment_model, metadata = get_model(device, config)

    if not os.path.exists(os.path.join(script_dir, "temp/")):
        os.makedirs(os.path.join(script_dir, "temp/"))

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video [{i+1}/{len(input_paths)}]\t{input_path}")

        out_loc = output_paths[i]

        if os.path.exists(os.path.join(out_loc, "asr_whisperx.pkl")):
            print("Already processed. Skipping...")
            continue

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        video_feat_dict = transcribe_video(script_dir, input_path, model, diarize_model, 
                                            alignment_model, metadata, device, config)

        with open(os.path.join(out_loc, "asr_whisperx.pkl"), "wb") as f:
            pickle.dump(video_feat_dict, f)

        print()


if __name__ == "__main__":
    main()