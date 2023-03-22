# #/******

# Author: Gullal
# Code: Runs spacy NLP and Flair pipeline for Named Entity Recognition (NER) on ASR transcript
# Input: Sentence or Multiple Sentences
# Output: List of Word_NE

# *******\

import os
import re
import pickle
from datetime import date
import numpy as np
import json
import csv
import jsonlines
import urllib.parse
from urllib.request import Request
from sortedcollections import OrderedSet

import spacy
## Github Repository: https://github.com/explosion/spaCy
## Commit ID: dec81508d28b47f09a06203c472b37f00db6c869
## POS Tag and NER List: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py

from flair.data import Sentence
from flair.models import SequenceTagger
## flair 0.11.3

## Other repos
## https://github.com/ambiverse-nlu/ambiverse-nlu

from entity_linking import link_annotations, fix_entity_types

import argparse

parser = argparse.ArgumentParser(description="Run NER, linking on ASR transcript and extract named entities")
parser.add_argument("-m", "--media", type=str, default="CompactTV", help=" CompactTV | BildTV | tagesschau")
args = parser.parse_args()


def get_wikifier_annotations(doc, all_sents, language="de", wikifier_key="dqmaycxjptujqkdwsjojeblwjdiovu"):
    threshold = 1.0
    endpoint = 'http://www.wikifier.org/annotate-article'
    language = language
    wikiDataClasses = 'false'
    wikiDataClassIds = 'true'
    includeCosines = 'false'
    all_annotations = []
    data = urllib.parse.urlencode([("text", doc.text.strip()[:24999]), ("lang", language), ("userKey", wikifier_key),
                                ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "true"),
                                ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
                                ("wikiDataClasses", wikiDataClasses), ("wikiDataClassIds", wikiDataClassIds),
                                ("support", "true"), ("ranges", "false"), ("includeCosines", includeCosines),
                                ("maxMentionEntropy", "3")])

    req = urllib.request.Request(endpoint, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout=60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
        if 'annotations' in response:
            for sent in all_sents:
                sent_annot = []
                for annot in response['annotations']:                 
                    for annot_occ in annot['support']:
                        if annot_occ['chFrom'] >= sent.start_char and annot_occ['chTo'] <= sent.end_char:
                            sent_annot.append(annot)
                        elif annot_occ['chTo'] > sent.end_char:
                            break
                all_annotations.append(sent_annot)
        else:
            print(f'No valid response: {response}')

    return all_annotations
                

# Sentence-wise NER
def get_spacy_flair_annotations(all_sents, tagger):
    named_entities = []
    
    tot_len = 0
    for sent in all_sents:
        sent_ents = []
        sent_ent_key = set()
        ## Get Spacy NEs
        for ent in sent.ents:
            sent_ent_key.add(ent.text.strip("."))
            sent_ents.append({
                'text': ent.text.strip("."),
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                })
            
        ## Get Flair NEs
        sentence = Sentence(sent.text)
        tagger.predict(sentence)

        for ent in sentence.get_spans('ner'):
            if ent.text.strip(".") not in sent_ent_key:
                for match in re.finditer(ent.text.strip("."), sent.text):
                    sent_ents.append({
                        'text': ent.text,
                        'type': ent.get_label("ner").value,
                        'start': match.start()+tot_len,
                        'end': match.end()+tot_len,
                    })

        tot_len += len(sent.text)

        named_entities.append(sent_ents)

    return named_entities


## spacy model
## Download the model with `python -m spacy download de_core_news_lg`
nlp = spacy.load("de_core_news_lg")

## Flair NER model
tagger = SequenceTagger.load("flair/ner-german-large")

## Locations
media = args.media
date_today = date.today().strftime("%Y%m%d")
print(date_today)

## Event list
eventKG = set()
with open("eventKG.csv", "r") as csvfile:
    content = csv.reader(csvfile)
    for row in content:
        eventKG.add(row[0])


## Location of transcripts
in_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

## Output location
out_loc = "/nfs/data/fakenarratives/%s/results/20230202/"%(media)

# in_loc = "results/%s"%(media)
# out_loc = "results/%s"%(media)

persons_dict = {}

for i, fldr in enumerate(os.listdir(in_loc)):

    video_feat_dict = { "github_repo": "https://github.com/explosion/spaCy;https://github.com/flairNLP/flair",
                        "commit_id": "dec81508d28b47f09a06203c472b37f00db6c869;d20c3fd07474e6103780fbe2b65302871f8db33d",
                        "other_libs": "de_core_news_lg-3.5.0-py3-none-any.whl",
                        "parameters": "default",
                        "video_file": os.path.join(in_loc, fldr+".mp4"),
                        "output_data": {},
                      }

    fname = os.path.join(in_loc, fldr, "asr.pkl")
    print("Video: ", i+1, fname) 
    
    asr_dict = pickle.load(open(fname,'rb'))

    doc = nlp(asr_dict["output_data"]["text"])

    all_texts = [sent for sent in doc.sents if len(sent.text) > 1]
    
    spacy_flair_nes = get_spacy_flair_annotations(all_texts, tagger)

    wikifier_nes = get_wikifier_annotations(doc, all_texts)

    linked_entities = link_annotations(spacy_flair_nes, wikifier_nes)

    print("Number of NEs: %d"%(len(linked_entities)))
    video_feat_dict["output_data"] = linked_entities

    pickle.dump(video_feat_dict, open(os.path.join(out_loc, fldr, "ner_linking.pkl"), "wb"))

    ## Saves all the persons into a dictionary and jsonl file
#     unk_cnt = 1
#     for sent_ents in linked_entities:
#         for ent in sent_ents:
#             if ent["type"] == "PER":
#                 if ent["disambiguation"]:
#                     persons_dict[ent["wd_id"]] = ent
#                 else:
#                     ent["wd_id"] = "unknown_%d"%(unk_cnt)
#                     persons_dict[ent["wd_id"]] = ent
#                     unk_cnt += 1
    

# print("Number of persons: %d"%(len(persons_dict)))


# with jsonlines.open("outputs/%s_persons.jsonl"%(media), "w") as fw:
#     for key in persons_dict:
#         fw.write(persons_dict[key])
