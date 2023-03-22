# #/******

# Author: Gullal
# Code: Runs keywords search on ASR transcript for filtering segments
# Input: ASR text transcript and segments
# Output: Segments marked with keyword match or not

# *******\

import os
import re
import pickle
from datetime import date
import numpy as np
import json
import csv
import jsonlines

import argparse

parser = argparse.ArgumentParser(description="Runs keywords search on ASR transcript for filtering segments")
parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
parser.add_argument("-k", "--keyfile", type=str, default="keywords.txt", help="txt file with a word in each line")
args = parser.parse_args()


## Locations
media = args.media
date_today = date.today().strftime("%Y%m%d")
print(date_today)

## Location of transcripts
in_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

## Output location
out_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

## Read in keywords
keywords  = set(word.strip().lower() for word in open(args.keyfile).readlines())

for i, fldr in enumerate(os.listdir(in_loc)):

    video_feat_dict = { "github_repo": "",
                        "commit_id": "",
                        "other_libs": "",
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fldr+".mp4"),
                        "matched": False,
                        "output_data": {},
                      }

    fname = os.path.join(in_loc, fldr, "asr.pkl")
    # print("Video: ", i+1, fname) 

    asr_dict = pickle.load(open(fname,'rb'))
    segments = asr_dict["output_data"]["segments"]

    matched_segments = []
    
    for seg in segments:
        text = re.sub(r"[,.;?!]+\ *", " ", seg["text"])
        words = set([word.strip().lower() for word in text.split() if len(word) > 2])

        temp = {"id": seg["id"], "seek": seg["seek"], "start": seg["start"], "end": seg["end"]}

        if words.intersection(keywords):
            temp["match"] = True
            video_feat_dict["matched"] = True
        else:
            temp["match"] = False
        
        matched_segments.append(temp)
    

    video_feat_dict["output_data"] = matched_segments

    pickle.dump(video_feat_dict, open(os.path.join(out_loc, fldr, "keyword_match.pkl"), "wb"))

    # if video_feat_dict["matched"]:
    #     print(fldr)


