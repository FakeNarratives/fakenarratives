import os
import pickle
import tempfile
from moviepy.editor import VideoFileClip
import torch
import whisper
import argparse
from pyannote.audio import Pipeline
import sys
sys.path.append("audio_analysis/pyannote-whisper/")
from pyannote_whisper.utils import diarize_text

def parse_args():
    parser = argparse.ArgumentParser(description="Runs OpenAI's whisper model on news videos for automatic speech recognition and speaker diarization")
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


def get_model(device):
    model = whisper.load_model("large-v2")
    model.to(device)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
    pipeline.to(device)

    return model, pipeline

def transcribe_video(video_path, model, pipeline):
    """
    Transcribes video using OpenAI's Whisper ASR model and stores details in a dictionary.
    Also performs speaker diarization.

    The dictionary includes:
    1. GitHub repo of the Whisper model
    2. Commit ID of the used Whisper model
    3. Parameters used (default in this case)
    4. Video file path
    5. Transcription result

    Args:
        video_path (str): Full path to the video file
        model (whisper.asr.ASRModel): Initialized Whisper ASR model

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (str): Transcription and diarization result from the Whisper model
    """

    video_feat_dict = {"github_repo": "https://github.com/openai/whisper;https://github.com/pyannote/pyannote-audio",
                        "commit_id": "f572f2161ba831bae131364c3bffdead7af6d210;f3935464065f743496df9558b7badaaa5a827c9c",
                        "parameters": "default",
                        "video_file": video_path,
                      }

    video = VideoFileClip(video_path+".mp4")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(os.path.join(script_dir, "temp/")):
        os.makedirs(os.path.join(script_dir, "temp/"))

    with tempfile.NamedTemporaryFile(os.path.join(script_dir, "temp/"), suffix=".mp3", delete=True) as tmp:
        video.audio.write_audiofile(tmp.name)
        asr_result = model.transcribe(tmp.name, word_timestamps=True)
        diarization_result = pipeline(tmp.name)

    final_result = diarize_text(asr_result, diarization_result)

    speaker_segments = []
    for seg, spk, sent in final_result:
        speaker_segments.append({"start": seg.start, "end": seg.end, "speaker": spk, "text": sent})

    asr_result["speaker_segments"] = speaker_segments

    video_feat_dict["output_data"] = asr_result

    return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    model, pipeline = get_model(device)

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path) 

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        video_feat_dict = transcribe_video(input_path, model, pipeline)

        with open(os.path.join(out_loc, "asr.pkl"), "wb") as f:
            pickle.dump(video_feat_dict, f)


if __name__ == "__main__":
    main()