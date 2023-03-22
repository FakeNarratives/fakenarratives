# #/******

# Author: Gullal
# Code: Runs OpenAI's whisper model on news videos for automatic speech recognition
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

import argparse

parser = argparse.ArgumentParser(description="Runs OpenAI's whisper model on news videos for automatic speech recognition")
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

## Load whisper model
model = whisper.load_model("large-v2")
model.to(device)

## Temp file for mp4 to mp3 audio
tmp = tempfile.NamedTemporaryFile(suffix=".mp3")

for i,fname in enumerate(os.listdir(in_loc)):
    print("Video: ", i+1, fname) 

    if not os.path.exists(os.path.join(out_loc, fname.replace(".mp4",""))):
        os.makedirs(os.path.join(out_loc, fname.replace(".mp4","")))

    video_feat_dict = {"github_repo": "https://github.com/openai/whisper",
                        "commit_id": "7858aa9c08d98f75575035ecd6481f462d66ca27",
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fname),
                      }

    video = VideoFileClip(os.path.join(in_loc, fname))
    
    video.audio.write_audiofile(tmp.name)

    result = model.transcribe(tmp.name)

    video_feat_dict["output_data"] = result

    pickle.dump(video_feat_dict, open(os.path.join(out_loc, fname.replace(".mp4",""), "asr.pkl"), "wb"))





    


