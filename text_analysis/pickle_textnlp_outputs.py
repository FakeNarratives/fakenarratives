import os
import pickle
import re
import numpy as np
import stanza
import torch
from germansentiment import SentimentModel
import sys
sys.path.append(".")
from text_analysis.ner_utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Runs german sentiment detection model on ASR transcript")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to video transcriptions as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for input transcriptions")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args

def read_transcript_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
        transcript_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in transcript_paths], [os.path.join(base_output_dir, vp) for vp in transcript_paths]


def get_model():
    nlp = stanza.Pipeline(lang='de', processors='tokenize,ner,pos', use_gpu=True)
    sen_model = SentimentModel('mdraw/german-news-sentiment-bert')
    return nlp, sen_model

def add_transcript_boundaries(asr_dict):    ## Segment boundaries are needed to match stanza NERs with Wikifier annotations
    ## Load ASR transcript and add start/end characters of segments
    end_char = 0
    cnt_i = 0
    cnt_j = 0
    for segment in asr_dict["output_data"]["segments"]:
        if segment["text"] != "": ## Ignore empty segments
            end_char += len(segment["text"])
            for match in re.finditer(re.escape(segment["text"]), asr_dict["output_data"]["text"]):
                if match.end() == end_char or match.end() == end_char-1:
                    segment["start_char"] = match.start()
                    segment["end_char"] = match.end()-1
                    cnt_j += 1
        
            cnt_i += 1

    assert(cnt_i==cnt_j)

    ## Load ASR transcript and add start/end characters of segments
    end_char = 0
    cnt_i = 0
    cnt_j = 0
    for segment in asr_dict["output_data"]["speaker_segments"]:
        if segment["text"] != "": ## Ignore empty segments
            end_char += len(segment["text"])
            for match in re.finditer(re.escape(segment["text"]), asr_dict["output_data"]["text"]):
                if match.end() == end_char or match.end() == end_char-1:
                    segment["start_char"] = match.start()
                    segment["end_char"] = match.end()-1
                    cnt_j += 1
        
            cnt_i += 1

    assert(cnt_i==cnt_j)

    return asr_dict


def get_sentiment(sen_model, asr_dict, proc_text):
    # Sentence-wise sentiment
    all_texts = [sent.text for sent in proc_text.sentences]
    preds, probs = sen_model.predict_sentiment(all_texts, output_probabilities=True)
    sent_sentiments = [{"id": str(i), "text": text, "pred": preds[i], "prob": [probs[i][0][1], probs[i][1][1], probs[i][2][1]]} for i, text in enumerate(all_texts)]

    # Segment-wise sentiment
    all_texts = [text_seg["text"].strip() for text_seg in asr_dict["output_data"]["segments"]]
    preds, probs = sen_model.predict_sentiment(all_texts, output_probabilities=True)
    seg_sentiments = [{"id": seg["id"], "seek": seg["seek"], "start": seg["start"], "end": seg["end"],
                    "pred": preds[i], "prob": [probs[i][0][1], probs[i][1][1], probs[i][2][1]]} 
                    for i, seg in enumerate(asr_dict["output_data"]["segments"])]

    # Speaker-Segment-wise sentiment
    all_texts = [text_seg["text"].strip() for text_seg in asr_dict["output_data"]["speaker_segments"]]
    preds, probs = sen_model.predict_sentiment(all_texts, output_probabilities=True)
    spk_sentiments = [{"id": str(i), "start": seg["start"], "end": seg["end"], "speaker": seg["speaker"],
                    "pred": preds[i], "prob": [probs[i][0][1], probs[i][1][1], probs[i][2][1]]} 
                    for i, seg in enumerate(asr_dict["output_data"]["speaker_segments"])]

    return sent_sentiments, seg_sentiments, spk_sentiments


def get_pos_tags(proc_text, proc_segments, proc_speaker_segments, asr_dict, pos_dict):
    ## Sentence-wise POS Tags
    sent_pos = []
    for i, sent in enumerate(proc_text.sentences):
        temp_pos = []
        pos_vector = np.zeros(14)
        for word in sent.words:
            temp_pos.append((word.text, word.upos, word.xpos, word.start_char, word.end_char))
            if word.upos in pos_dict:
                pos_vector[pos_dict[word.upos]] += 1
            else:
                pos_vector[pos_dict['X']] += 1
    
        sent_pos.append({"id": str(i), "text": sent.text, "tags": temp_pos, "vector": pos_vector})

    ## Segment-wise POS Tags
    seg_pos = []
    for seg_doc, seg in zip(proc_segments, asr_dict["output_data"]["segments"]):
        temp_pos = []
        pos_vector = np.zeros(14)
        for sent in seg_doc.sentences:    
            for word in sent.words:
                temp_pos.append((word.text, word.upos, word.xpos, word.start_char, word.end_char))
                if word.upos in pos_dict:
                    pos_vector[pos_dict[word.upos]] += 1
                else:
                    pos_vector[pos_dict['X']] += 1
    
        seg_pos.append({"id": seg["id"], "seek": seg["seek"], "start": seg["start"], 
                        "end": seg["end"], "tags": temp_pos, "vector": pos_vector})
    

    ## Speaker-Segment-wise POS Tags
    speaker_seg_pos = []
    for i, tupl in enumerate(zip(proc_speaker_segments, asr_dict["output_data"]["speaker_segments"])):
        seg_doc, seg = tupl
        temp_pos = []
        pos_vector = np.zeros(14)
        for sent in seg_doc.sentences:    
            for word in sent.words:
                temp_pos.append((word.text, word.upos, word.xpos, word.start_char, word.end_char))
                if word.upos in pos_dict:
                    pos_vector[pos_dict[word.upos]] += 1
                else:
                    pos_vector[pos_dict['X']] += 1
    
        speaker_seg_pos.append({"id": str(i), "start": seg["start"], "end": seg["end"], 
                                "speaker": seg["speaker"], "tags": temp_pos, "vector": pos_vector})


    return sent_pos, seg_pos, speaker_seg_pos


def get_ner_tags(proc_text, proc_segments, proc_speaker_segments, asr_dict, ner_dict, event_set):
    stanza_nes = get_stanza_ner_annotations(proc_text)
    wikifier_nes = get_wikifier_annotations(proc_text)
    linked_entities = link_annotations(stanza_nes, wikifier_nes)
    linked_entities = fix_entity_types(linked_entities, event_set)

    ent_map = {}
    for ent in linked_entities:
        ent_map[ent["text"]] = ent

    ## Sent-wise ent vectors
    sent_ent_vectors = []
    for i, sent in enumerate(proc_text.sentences):
        temp_nes = []
        ner_vector = np.zeros(6)
        for ent in sent.ents:
            if ent.text in ent_map:
                temp_nes.append((ent.text, ent_map[ent.text]["wd_label"], ent_map[ent.text]["type"], 
                                 ent_map[ent.text]["wd_id"], ent.start_char, ent.end_char))
                ner_vector[ner_dict[ent_map[ent.text]["type"]]] += 1

        sent_ent_vectors.append({"id": str(i), "text": sent.text, "tags": temp_nes, "vector": ner_vector})

    ## Segment-wise ent vectors
    seg_ent_vectors = []
    for seg_doc, seg in zip(proc_segments, asr_dict["output_data"]["segments"]):
        temp_nes = []
        ner_vector = np.zeros(6)
        for sent in seg_doc.sentences:
            for ent in sent.ents:
                if ent.text in ent_map:
                    temp_nes.append((ent.text, ent_map[ent.text]["wd_label"], ent_map[ent.text]["type"], 
                                     ent_map[ent.text]["wd_id"], ent.start_char, ent.end_char))
                    ner_vector[ner_dict[ent_map[ent.text]["type"]]] += 1

        seg_ent_vectors.append({"id": seg["id"], "seek": seg["seek"], "start": seg["start"], 
                                "end": seg["end"], "tags": temp_nes, "vector": ner_vector})

    
    ## Speaker-Segment-wise ent vectors
    speaker_seg_ent_vectors = []
    for i, tupl in enumerate(zip(proc_speaker_segments, asr_dict["output_data"]["speaker_segments"])):
        seg_doc, seg = tupl
        temp_nes = []
        ner_vector = np.zeros(6)
        for sent in seg_doc.sentences:
            for ent in sent.ents:
                if ent.text in ent_map:
                    temp_nes.append((ent.text, ent_map[ent.text]["wd_label"], ent_map[ent.text]["type"], 
                                     ent_map[ent.text]["wd_id"], ent.start_char, ent.end_char))
                    ner_vector[ner_dict[ent_map[ent.text]["type"]]] += 1

        speaker_seg_ent_vectors.append({"id": str(i), "start": seg["start"], "end": seg["end"], 
                                        "speaker": seg["speaker"], "tags": temp_nes, "vector": ner_vector})


    return sent_ent_vectors, seg_ent_vectors, speaker_seg_ent_vectors

    
def perform_nlp(video_path, asr_dict, nlp, 
                sen_model, pos_dict, ner_dict, 
                event_set):
    """
    Takes ASR segments and performs NLP of text and text segments for sentiment, POS and NER

    Args:
        video_path (str): Full path to the video file
        asr_dict (dict): ASR output dictionary
        nlp (stanza.Pipeline): Stanza NLP pipeline
        sen_model (germansentiment.SentimentModel): German sentiment detection model
        pos_dict (dict): Dictionary with pos tag label map
        ner_dict (dict): Dictionary with ner label map
        event_set (set): Set of events to be considered for NER

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): Dictionary containing sentiment, pos and ner outputs 
                                for sentence-wise, segment-wise and speaker-segment-wise segments
    """
    video_feat_dict = {"github_repo": "https://github.com/oliverguhr/german-sentiment-lib;https://github.com/stanfordnlp/stanza",
                        "commit_id": "367f8f55d92fd85e6cde8bc59dc8dbad7ec88071;a85cce6816f40fa03ab06f6497c7d65ba1244a33",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {"sentiment": {"sentence_wise": {}, "segment_wise": {}, "speakerseg_wise": {}},
                                        "pos": {"sentence_wise": {}, "segment_wise": {}, "speakerseg_wise": {}},
                                        "ner": {"sentence_wise": {}, "segment_wise": {}, "speakerseg_wise": {}}}
                        }
    
    proc_text = nlp(asr_dict["output_data"]["text"])
    proc_segments = [nlp(segment["text"]) for segment in asr_dict["output_data"]["segments"]]
    proc_speaker_segments = [nlp(segment["text"]) for segment in asr_dict["output_data"]["speaker_segments"]]
    
    ##------------------------------------------------Sentiment Analysis------------------------------------------------##
    t1 = time.time()
    
    sent_sentiments, seg_sentiments, spk_sentiments = get_sentiment(sen_model, asr_dict, proc_text)
    video_feat_dict["output_data"]["sentiment"]["sentence_wise"] = sent_sentiments
    video_feat_dict["output_data"]["sentiment"]["segment_wise"] = seg_sentiments
    video_feat_dict["output_data"]["sentiment"]["speakerseg_wise"] = spk_sentiments

    print("Sentiment Analysis Done in %.2f seconds"%(time.time()-t1))
    ##------------------------------------------------Sentiment Analysis------------------------------------------------##

    ##------------------------------------------------Parts of Speech------------------------------------------------##
    t2 = time.time()

    sent_pos, seg_pos, speaker_seg_pos = get_pos_tags(proc_text, proc_segments, proc_speaker_segments, asr_dict, pos_dict)
    video_feat_dict["output_data"]["pos"]["sentence_wise"] = sent_pos
    video_feat_dict["output_data"]["pos"]["segment_wise"] = seg_pos
    video_feat_dict["output_data"]["pos"]["speakerseg_wise"] = speaker_seg_pos    

    print("POS Tags Done in %.2f seconds"%(time.time()-t2))
    ##------------------------------------------------Parts of Speech------------------------------------------------##

    ##------------------------------------------------Named Entity Recognition------------------------------------------------##
    t3 = time.time()

    sent_ent_vectors, seg_ent_vectors, speaker_seg_ent_vectors = get_ner_tags(proc_text, proc_segments, proc_speaker_segments, 
                                                                              asr_dict, ner_dict, event_set)
    video_feat_dict["output_data"]["ner"]["sentence_wise"] = sent_ent_vectors
    video_feat_dict["output_data"]["ner"]["segment_wise"] = seg_ent_vectors
    video_feat_dict["output_data"]["ner"]["speakerseg_wise"] = speaker_seg_ent_vectors

    print("Named Entity Recognition Done in %.2f seconds"%(time.time()-t3))

    ##------------------------------------------------Named Entity Recognition------------------------------------------------##

    return video_feat_dict



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_dict = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CONJ": 4, "CCONJ": 4, "SCONJ": 4, "DET": 5, "INTJ": 6, 
            "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "VERB": 12, "X": 13, "PUNCT": 13, "SYM": 13}

    ner_dict = {"EPER": 0, "LPER": 1, "LOC": 2, "ORG": 3, "EVENT": 4, "MISC": 5}

    event_set = set()
    with open("text_analysis/eventKG.csv") as fr:
        for line in fr:
            event_set.add(line.strip())

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_transcript_paths(args.file, args.inp_dir, args.out_dir)

    ## Get nlp and sentiment models
    nlp, sen_model = get_model()

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path)

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "asr_textnlp.pkl")):
            asr_dict = pickle.load(open(os.path.join(input_path, "asr.pkl"), "rb"))
            asr_dict = add_transcript_boundaries(asr_dict)
            video_path = input_path.replace("results_pkl", "videos")

            video_feat_dict = perform_nlp(video_path, asr_dict, nlp, 
                                            sen_model, pos_dict, ner_dict, event_set)
                
            with open(os.path.join(out_loc, "asr_textnlp.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)


if __name__ == "__main__":
    main()
