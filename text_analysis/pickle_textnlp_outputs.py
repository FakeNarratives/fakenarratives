import os
import pickle
import re
import numpy as np
import stanza
import torch
from germansentiment import SentimentModel
from text_utils import *
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Runs german sentiment detection model, POS Tagging and NEL on ASR transcript")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to video transcriptions as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for input transcriptions")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    parser.add_argument("-r", "--rewrite", action="store_true", help="Rewrite existing files")
    args = parser.parse_args()
    return args


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def read_transcript_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
        transcript_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in transcript_paths], [os.path.join(base_output_dir, vp) for vp in transcript_paths]


def get_model():
    nlp = stanza.Pipeline(lang='de', processors='tokenize,ner,pos', use_gpu=True)
    sen_model = SentimentModel('mdraw/german-news-sentiment-bert')
    return nlp, sen_model


def perform_ner(video_path, speaker_turns, nlp, ner_dict, event_set):
    """
    Takes ASR segments and performs NER on text segments

    Args:
        video_path (str): Full path to the video file
        segments (list): List of ASR segments
        nlp (stanza.Pipeline): Stanza NLP pipeline
        ner_dict (dict): Dictionary mapping NER tags to indices
        event_set (set): Set of events from EventKG

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): Dictionary containing NER outputs for speaker segment-wise and speaker-turn segments
    """
    video_feat_dict = {"github_repo": "https://github.com/stanfordnlp/stanza",
                        "commit_id": "a85cce6816f40fa03ab06f6497c7d65ba1244a33",
                        "parameters": "default",
                        "video_file":  video_path.replace("results_pkl", "videos"),
                        "output_data": {"speakerturn_wise": {}, "ner_labelmap": ner_dict}
                        }
    
    ## Speaker-turn wise NER Tags
    speaker_turn_ner = []
    for segment in speaker_turns:
        if segment["end"] <= segment["start"]:
            speaker_turn_ner.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                      "tags": None, "vector": None})
            continue
        
        ner_vector = np.zeros(6)

        stanza_nes = get_stanza_ner_annotations(nlp(segment["text"]))
        wikifier_nes = get_wikifier_annotations(segment["text"])
        linked_entities = link_annotations(stanza_nes, wikifier_nes)
        linked_entities = fix_entity_types(linked_entities, event_set)

        for ent in linked_entities:
            ner_vector[ner_dict[ent["type"]]] += 1

        speaker_turn_ner.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                "tags": linked_entities, "vector": ner_vector})

    video_feat_dict["output_data"]["speakerturn_wise"] = speaker_turn_ner

    return video_feat_dict


def perform_sentiment(video_path, speaker_turns, nlp, sen_model, sent_dict):
    """
    Takes ASR segments and performs sentiment analysis of text segments

    Args:
        video_path (str): Full path to the video file
        speaker_segments (list): List of ASR segments
        sen_model (germansentiment.SentimentModel): German sentiment detection model
        sent_dict (dict): Dictionary mapping sentiment labels to indices

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): Dictionary containing sentiment outputs for sentence-wise, segment-wise and speaker-segment-wise segments
    """
    video_feat_dict = {"github_repo": "https://github.com/stanfordnlp/stanza",
                        "commit_id": "a85cce6816f40fa03ab06f6497c7d65ba1244a33",
                        "parameters": "default",
                        "video_file":  video_path.replace("results_pkl", "videos"),
                        "output_data": {"sentence_wise": {}, "speakerturn_wise": {}, "sent_labelmap": sent_dict}
                        }

    sentwise_sentiments = []
    spkturn_sentiments = []
    for segment in speaker_turns:
        if segment["end"] <= segment["start"]:
            sentwise_sentiments.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                      "vector": None})
            spkturn_sentiments.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                      "pred": None, "prob": None})
            continue
        
        ## Sentence-wise sentiment proportion in speaker turn
        doc = nlp(segment["text"].strip())
        turn_sentences = [sent.text for sent in doc.sentences]
        # if len(turn_sentences) == 0:
        #     continue
        preds, _ = sen_model.predict_sentiment(turn_sentences, output_probabilities=True)
        ## Aggregate proportion of sentiments over sentences
        sent_vector = np.zeros(3)
        for i, pred in enumerate(preds):
            sent_vector[sent_dict[pred]] += 1

        sent_vector = sent_vector / len(turn_sentences)
    
        sentwise_sentiments.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                    "vector": sent_vector})
    
        # Speaker-Turn-wise sentiment
        st_preds, st_probs = sen_model.predict_sentiment([segment["text"].strip()], output_probabilities=True)
        
        spkturn_sentiments.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                         "pred": st_preds[0], "prob": [st_probs[0][0][1], st_probs[0][1][1], st_probs[0][2][1]]})
    
    video_feat_dict["output_data"]["sentence_wise"] = sentwise_sentiments
    video_feat_dict["output_data"]["speakerturn_wise"] = spkturn_sentiments

    return video_feat_dict


def perform_pos(video_path, speaker_turns, nlp, pos_dict):
    """
    Takes ASR segments and performs POS tagging of text segments

    Args:
        video_path (str): Full path to the video file
        segments (list): List of ASR segments
        nlp (stanza.Pipeline): Stanza NLP pipeline
        pos_dict (dict): Dictionary mapping POS tags to indices

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): Dictionary containing POS outputs for sentence-wise, segment-wise and speaker-segment-wise segments
    """
    video_feat_dict = {"github_repo": "https://github.com/stanfordnlp/stanza",
                        "commit_id": "a85cce6816f40fa03ab06f6497c7d65ba1244a33",
                        "parameters": "default",
                        "video_file":  video_path.replace("results_pkl", "videos"),
                        "output_data": {"speakerturn_wise": {}, "pos_labelmap": pos_dict}
                        }
    
    # ## Speaker-Segment-wise POS Tags
    # speaker_seg_pos = []
    # for segment in speaker_segments:
    #     seg_doc = nlp(segment["text"])
    #     temp_pos = []
    #     pos_vector = np.zeros(14)
    #     for sent in seg_doc.sentences:    
    #         for word in sent.words:
    #             temp_pos.append((word.text, word.upos, word.xpos, word.start_char, word.end_char))
    #             if word.upos in pos_dict:
    #                 pos_vector[pos_dict[word.upos]] += 1
    #             else:
    #                 pos_vector[pos_dict['X']] += 1
    
    #     speaker_seg_pos.append({"start": segment["start"], "end": segment["end"],
    #                             "speaker": segment["speaker"] if "speaker" in segment else "Unknown", 
    #                             "tags": temp_pos, "vector": pos_vector})
    # video_feat_dict["output_data"]["speakerseg_wise"] = speaker_seg_pos
    
    ## Speaker-Turn-wise POS Tags
    spkturn_pos = []
    for segment in speaker_turns:
        if segment["end"] <= segment["start"]:
            spkturn_pos.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                      "tags": None, "vector": None})
            continue
        seg_doc = nlp(segment["text"])
        temp_pos = []
        pos_vector = np.zeros(14)
        for sent in seg_doc.sentences:    
            for word in sent.words:
                temp_pos.append((word.text, word.upos, word.xpos, word.start_char, word.end_char))
                if word.upos in pos_dict:
                    pos_vector[pos_dict[word.upos]] += 1
                else:
                    pos_vector[pos_dict['X']] += 1
    
        spkturn_pos.append({"start": segment["start"], "end": segment["end"],
                                "speaker": segment["speaker"], "tags": temp_pos, "vector": pos_vector})
        

    video_feat_dict["output_data"]["speakerturn_wise"] = spkturn_pos

    return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seeds(42)

    sent_dict = {"positive": 0, "negative": 1, "neutral": 2}

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
        print(f"Processing Video [{i+1}/{len(input_paths)}]\t{input_path}")
        
        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not args.rewrite:
            print("Already processed. Skipping...")
            continue

        # whisper_segments = pickle.load(open(os.path.join(input_path, "asr_whisper.pkl"), "rb"))["output_data"]["speaker_segments"]
        whisperx_segments = pickle.load(open(os.path.join(input_path, "asr_whisperx.pkl"), "rb"))["output_data"]["speaker_segments"]
        # asr_dict = add_transcript_boundaries(asr_dict)
        
        # speaker_turns = get_speaker_turns(whisper_segments)
        speaker_turns = get_speaker_turns(whisperx_segments)

        # sentiment_feat_dict = perform_sentiment(input_path, speaker_turns, nlp, sen_model, sent_dict)
        sentiment_feat_dict2 = perform_sentiment(input_path, speaker_turns, nlp, sen_model, sent_dict)
        
        print("Number of sentiment features:", len(sentiment_feat_dict2["output_data"]["speakerturn_wise"]), ", Number of speaker turns:", len(speaker_turns))
        assert len(sentiment_feat_dict2["output_data"]["speakerturn_wise"]) == len(speaker_turns)

        # pos_feat_dict = perform_pos(input_path, speaker_turns, nlp, pos_dict)
        pos_feat_dict2 = perform_pos(input_path, speaker_turns, nlp, pos_dict)
        
        print("Number of PoS features:", len(pos_feat_dict2["output_data"]["speakerturn_wise"]), ", Number of speaker turns:", len(speaker_turns))
        assert len(pos_feat_dict2["output_data"]["speakerturn_wise"]) == len(speaker_turns)

        # ner_feat_dict = perform_ner(input_path, speaker_turns, nlp, ner_dict, event_set)
        ner_feat_dict2 = perform_ner(input_path, speaker_turns, nlp, ner_dict, event_set)
        
        print("Number of NER features:", len(ner_feat_dict2["output_data"]["speakerturn_wise"]), ", Number of speaker turns:", len(speaker_turns))
        assert len(ner_feat_dict2["output_data"]["speakerturn_wise"]) == len(speaker_turns)

        # print("Saving to", os.path.join(out_loc, "whisper_sentiment.pkl"))
        # with open(os.path.join(out_loc, "whisper_sentiment.pkl"), "wb") as output_file:
        #     pickle.dump(sentiment_feat_dict, output_file)
        
        print("Saving to", os.path.join(out_loc, "whisperx_sentiment.pkl"))
        with open(os.path.join(out_loc, "whisperx_sentiment.pkl"), "wb") as output_file:
            pickle.dump(sentiment_feat_dict2, output_file)

        # print("Saving to", os.path.join(out_loc, "whisper_pos.pkl"))
        # with open(os.path.join(out_loc, "whisper_pos.pkl"), "wb") as output_file:
        #     pickle.dump(pos_feat_dict, output_file)
        
        print("Saving to", os.path.join(out_loc, "whisperx_pos.pkl"))
        with open(os.path.join(out_loc, "whisperx_pos.pkl"), "wb") as output_file:
            pickle.dump(pos_feat_dict2, output_file)

        # print("Saving to", os.path.join(out_loc, "whisper_ner.pkl"))
        # with open(os.path.join(out_loc, "whisper_ner.pkl"), "wb") as output_file:
        #     pickle.dump(ner_feat_dict, output_file)
        
        print("Saving to", os.path.join(out_loc, "whisperx_ner.pkl"))
        with open(os.path.join(out_loc, "whisperx_ner.pkl"), "wb") as output_file:
            pickle.dump(ner_feat_dict2, output_file)

        print()


if __name__ == "__main__":
    main()
