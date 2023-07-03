import os
import pickle
import tempfile
from datetime import date
from moviepy.editor import VideoFileClip
import torch
import whisper
## Github repository: https://github.com/openai/whisper
## Commit ID: f572f2161ba831bae131364c3bffdead7af6d210
import argparse

import os
import pickle
import tempfile
from datetime import date
from moviepy.editor import VideoFileClip
import torch
import whisper
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Runs OpenAI's whisper model on news videos for automatic speech recognition")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args

def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]

def get_output_dir(video_path, base_output_dir):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(base_output_dir, filename)
    return output_dir

def get_model(device):
    model = whisper.load_model("large-v2")
    model.to(device)
    return model

def transcribe_video(video_path, model):
    """
    Transcribes video using OpenAI's Whisper ASR model and stores details in a dictionary.

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
            github_repo (str): GitHub repository of the Whisper model
            commit_id (str): Commit ID of the used Whisper model
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (str): Transcription result from the Whisper model
    """

    video_feat_dict = {"github_repo": "https://github.com/openai/whisper",
                        "commit_id": "f572f2161ba831bae131364c3bffdead7af6d210",
                        "parameters": "default",
                        "video_file": video_path,
                      }

    video = VideoFileClip(video_path+".mp4")

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        video.audio.write_audiofile(tmp.name)
        result = model.transcribe(tmp.name)
        video_feat_dict["output_data"] = result

        return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    model = get_model(device)

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path) 

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        video_feat_dict = transcribe_video(input_path, model)

        fname = os.path.basename(input_path)

        with open(os.path.join(out_loc, "asr.pkl"), "wb") as f:
            pickle.dump(video_feat_dict, f)


if __name__ == "__main__":
    main()





# parser = argparse.ArgumentParser(description="Runs OpenAI's whisper model on news videos for automatic speech recognition")
# parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
# args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# media = args.media
# date_today = date.today().strftime("%Y%m%d")
# print(date_today)

# ## Location of videos
# if 'tage' not in media:
#     in_loc = "/nfs/data/fakenarratives/%s/videos/"%(media)
# else:
#     in_loc = "/nfs/data/fakenarratives/%s/videos/2022"%(media)

# ## Output location
# out_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

# ## Load whisper model
# model = whisper.load_model("large-v2")
# model.to(device)

# ## Temp file for mp4 to mp3 audio
# tmp = tempfile.NamedTemporaryFile(suffix=".mp3")

# for i,fname in enumerate(os.listdir(in_loc)):
#     print("Video: ", i+1, fname) 

#     if not os.path.exists(os.path.join(out_loc, fname.replace(".mp4",""))):
#         os.makedirs(os.path.join(out_loc, fname.replace(".mp4","")))

#     video_feat_dict = {"github_repo": "https://github.com/openai/whisper",
#                         "commit_id": "f572f2161ba831bae131364c3bffdead7af6d210",
#                         "parameters": "default",
#                         "video_file": os.path.join(in_loc, fname),
#                       }

#     video = VideoFileClip(os.path.join(in_loc, fname))
    
#     video.audio.write_audiofile(tmp.name)

#     result = model.transcribe(tmp.name)

#     video_feat_dict["output_data"] = result

#     pickle.dump(video_feat_dict, open(os.path.join(out_loc, fname.replace(".mp4",""), "asr.pkl"), "wb"))





    


