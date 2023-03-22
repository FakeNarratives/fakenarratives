# #/******

# Author: Gullal
# Code: Runs spacy NLP pipeline for POS-tagging on ASR transcript
# Input: Sentence or Multiple Sentences
# Output: List of Tuples of Word_POS_Tag

# *******\

import os
import pickle
from datetime import date
import numpy as np

import spacy
## Github Repository: https://github.com/explosion/spaCy
## Commit ID: 3d0e895363921d4acb7f89a5b708472681e6fc1b

import argparse

parser = argparse.ArgumentParser(description="Run NER, linking on ASR transcript and extract named entities")
parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
args = parser.parse_args()

## spacy model
nlp = spacy.load("de_core_news_lg")

## Locations
media = args.media
date_today = date.today().strftime("%Y%m%d")
print(date_today)

## Location of transcripts
in_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

## Output location
out_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

# in_loc = "results/%s"%(media)
# out_loc = "results/%s"%(media)

for i, fldr in enumerate(os.listdir(in_loc)):

    video_feat_dict = { "github_repo": "https://github.com/explosion/spaCy",
                        "commit_id": "dec81508d28b47f09a06203c472b37f00db6c869",
                        "other_libs": "de_core_news_lg-3.4.0-py3-none-any.whl",
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fldr+".mp4"),
                        "output_data": {},
                      }

    fname = os.path.join(in_loc, fldr, "asr.pkl")
    print("Video: ", i+1, fname) 
    
    asr_dict = pickle.load(open(fname,'rb'))
    
    doc = nlp(asr_dict["output_data"]["text"])
    
    ## Sentence-wise POS Tags
    all_pos = []
    for sent in doc.sents:
        sent_pos = []
        if len(sent.text) > 1:
            for token in sent:
                sent_pos.append((token.text, token.pos_, token.tag_))

            all_pos.append(sent_pos)

    print(len(all_pos))
    video_feat_dict["output_data"] = all_pos

    pickle.dump(video_feat_dict, open(os.path.join(out_loc, fldr, "pos_tagger.pkl"), "wb"))

    