# #/******

# Author: Gullal
# Code: Runs topic modelling on ASR transcript
# Input: ASR text transcript and segments
# Output: Top topics

# *******\
import os
import re
import pickle
from datetime import date
import numpy as np
import json
import csv
import jsonlines
import random

import torch
from transformers import pipeline

from bertopic import BERTopic
## Github repository: https://github.com/MaartenGr/BERTopic/
## Commit ID: 7142ce7f9c12e92ec9146ea97d8d4d5ef8332354

## Other repos
# https://huggingface.co/chkla/parlbert-topic-german
# chkla/parlbert-topic-german
# pipeline_classification_topics = pipeline("text-classification", model="chkla/parlbert-topic-german", tokenizer="bert-base-german-cased", return_all_scores=False)

import argparse

parser = argparse.ArgumentParser(description="Runs topic modelling on ASR transcript")
parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
args = parser.parse_args()

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

## Locations
media = args.media
date_today = date.today().strftime("%Y%m%d")
print(date_today)

## Location of transcripts
in_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

## Output location
out_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

## Get stopwords
stopwords = set([word.strip() for word in open("stopwords.txt").readlines()])

topic_model = BERTopic(language="multilingual")

all_texts = []
file_names = []
for i, fldr in enumerate(os.listdir(in_loc)):

    video_feat_dict = { "github_repo": "https://github.com/MaartenGr/BERTopic/",
                        "commit_id": "7142ce7f9c12e92ec9146ea97d8d4d5ef8332354",
                        "other_libs": "",
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fldr+".mp4"),
                        "output_data": {},
                      }

    fname = os.path.join(in_loc, fldr, "asr.pkl")
    print("Video: ", i+1, fname) 

    asr_dict = pickle.load(open(fname,'rb'))
    text = asr_dict["output_data"]["text"]

    text = " ".join([word.strip(".,?!") for word in text.split() if word.strip(".,?!").lower() not in stopwords])

    all_texts.append(text)
    file_names.append(fldr)

topic_model.fit_transform(all_texts)

print(topic_model.get_topic_info())

## The -1 topic refers to all outlier documents and are typically ignored.
print(topic_model.get_topic(-1))

print(topic_model.get_topic(0))
print(topic_model.get_topic(1))

topic_df = topic_model.get_document_info(all_texts)
topic_df["Video"] = file_names

topic_df.to_csv("outputs/%s_topics.tsv"%(media), sep="\t", index=None, columns=["Video", "Topic", "Name", "Top_n_words", "Probability", "Representative_document"])

