# #/******

# Author: Gullal
# Code: Runs OpenAI's whisper model on news videos for automatic speech recognition 
#           and additionally gets word time stamps
# Input: Video
# Output: Text

# *******\

import os
import pickle
import tempfile
from datetime import date

from moviepy.editor import VideoFileClip
import torch

import whisper
## Github repository: https://github.com/openai/whisper
## Commit ID: 7858aa9c08d98f75575035ecd6481f462d66ca27

import whisper_timestamped as whisper
## Github repository: https://github.com/linto-ai/whisper-timestamped
## Commit ID: ceadee8a22713d69972e6acc320aeeeab19427c1

import argparse

parser = argparse.ArgumentParser(description="Runs OpenAI's whisper model on news videos for ASR and word timestamps")
parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

media = args.media
date_today = date.today().strftime("%Y%m%d")
print(date_today)

## Location of videos
if 'tage' not in media:
    in_loc = "/nfs/data/fakenarratives/%s/videos/"%(media)
else:
    in_loc = "/nfs/data/fakenarratives/%s/videos/2022"%(media)

## Output location
out_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

# in_loc = "videos_for_bernhard/%s"%(media)
# out_loc = "results/%s"%(media)

## Load whisper model
model = whisper.load_model("large-v2", device=device)

## Temp file for mp4 to mp3 audio
tmp = tempfile.NamedTemporaryFile(suffix=".mp3")

for i, fname in enumerate(os.listdir(in_loc)):
    print("Video: ", i+1, fname)
    
    if not os.path.exists(os.path.join(out_loc, fname.replace(".mp4",""))):
        os.makedirs(os.path.join(out_loc, fname.replace(".mp4","")))

    video_feat_dict = {"github_repo": "https://github.com/linto-ai/whisper-timestamped",
                        "commit_id": "ceadee8a22713d69972e6acc320aeeeab19427c1",
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fname),
                      }

    video = VideoFileClip(os.path.join(in_loc, fname))
    
    video.audio.write_audiofile(tmp.name)

    audio = whisper.load_audio(tmp.name)

    result = whisper.transcribe(model, audio, language="de")

    video_feat_dict["output_data"] = result["segments"]

    pickle.dump(video_feat_dict, open(os.path.join(out_loc, fname.replace(".mp4",""), "asr_with_word_timestamps.pkl"), "wb"))