# #/******

# Author: Gullal
# Code: Runs sentiment detection model on whisper ASR transcript
# Input: Sentence or Multiple Sentences
# Output: Probability vector [Positive, Negative, Neutral]

# *******\

import os
import pickle
from datetime import date

import spacy

from germansentiment import SentimentModel
## Huggingface Link: https://huggingface.co/mdraw/german-news-sentiment-bert
## Huggingface Commit ID: 7b4abebe1c3fcfbc62dc0435e480807a80c18210
## Github Repository: https://github.com/text-analytics-20/news-sentiment-development
## Github Commit ID: 8209f0dcff8a07384aa1db684687b7133f1d608c

import argparse

parser = argparse.ArgumentParser(description="Runs german sentiment detection model on ASR transcript")
parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
args = parser.parse_args()

## spacy model
nlp = spacy.load("de_core_news_lg")


## Locations
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

## Load sentiment model
model = SentimentModel('mdraw/german-news-sentiment-bert')

for i, fldr in enumerate(os.listdir(in_loc)):

    video_feat_dict = {"github_repo": "https://github.com/text-analytics-20/news-sentiment-development",
                        "commit_id": "8209f0dcff8a07384aa1db684687b7133f1d608c",
                        "huggingface_repo": "https://huggingface.co/mdraw/german-news-sentiment-bert",
                        "huggingface_commit_id": "7b4abebe1c3fcfbc62dc0435e480807a80c18210",
                        "other_libs": "spacy:%s"%(spacy.__version__),
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fldr+".mp4"),
                        "output_data": {"sentence_wise": {}, "segment_wise": {}}
                      }

    fname = os.path.join(in_loc, fldr, "asr.pkl")
    print(fname) 
    
    asr_dict = pickle.load(open(fname,'rb'))
    
    doc = nlp(asr_dict["output_data"]["text"])
    
    ## Sentence-wise sentiment
    all_texts = []
    sentiments = []
    for sent in doc.sents:
        if len(sent.text) > 1:
            all_texts.append(sent.text.strip())

    preds, probs = model.predict_sentiment(all_texts, output_probabilities=True)

    for i, text in enumerate(all_texts):
        sentiments.append((text, preds[i], [probs[i][0][1], probs[i][1][1], probs[i][2][1]]))

    video_feat_dict["output_data"]["sentence_wise"] = sentiments

    ## Segment-wise sentiment
    all_texts = []
    sentiments = []
    for text_seg in asr_dict["output_data"]["segments"]:
        all_texts.append(text_seg["text"].strip())

    preds, probs = model.predict_sentiment(all_texts, output_probabilities=True)

    for i, text in enumerate(all_texts):
        sentiments.append((text, preds[i], [probs[i][0][1], probs[i][1][1], probs[i][2][1]]))

    video_feat_dict["output_data"]["segment_wise"] = sentiments

    pickle.dump(video_feat_dict, open(os.path.join(out_loc, fldr, "text_sentiment.pkl"), "wb"))