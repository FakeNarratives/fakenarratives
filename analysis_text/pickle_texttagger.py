import os
import pickle
import re
import numpy as np
import stanza
import torch
from text_utils import *
import argparse
import random
from pathlib import Path
import sys
sys.path.append(".")
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Runs POS Tagging and NEL on ASR transcript")
    parser.add_argument(
        "-v",
        "--videos",
        nargs="+",
        type=str,
        required=True,
        help="Path to input videos"
    )
    parser.add_argument(
        "-o",
        "--pkl_dir",
        type=str,
        required=True,
        help="Path to pkl directory"
    )
    parser.add_argument(
        "-r",
        "--rewrite",
        action="store_true",
        help="Rewrite existing files"
    )
    args = parser.parse_args()
    return args



def perform_ner(video_path, speaker_turns, nlp, ner_dict, org_list, event_list):
    """
    Takes ASR segments and performs NER on text segments

    Args:
        video_path (str): Full path to the video file
        segments (list): List of ASR segments
        nlp (stanza.Pipeline): Stanza NLP pipeline
        ner_dict (dict): Dictionary mapping NER tags to indices

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
                        "video_file":  video_path,
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

        stanza_nes = get_stanza_ner_annotations(nlp(segment["text"][:24999]))
        wikifier_nes = get_wikifier_annotations(segment["text"][:24999])
        linked_entities = link_annotations(stanza_nes, wikifier_nes)
        linked_entities = fix_entity_types(linked_entities, org_list, event_list)

        for ent in linked_entities:
            ner_vector[ner_dict[ent["type"]]] += 1

        speaker_turn_ner.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                "tags": linked_entities, "vector": ner_vector})
            
    video_feat_dict["output_data"]["speakerturn_wise"] = speaker_turn_ner

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
                        "video_file":  video_path,
                        "output_data": {"speakerturn_wise": {}, "pos_labelmap": pos_dict}
                        }
    
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

    pos_dict = {"ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CONJ": 4, "CCONJ": 4, "SCONJ": 4, "DET": 5, "INTJ": 6, 
            "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10, "PROPN": 11, "VERB": 12, "X": 13, "PUNCT": 13, "SYM": 13}

    ner_dict = {"EPER": 0, "LPER": 1, "LOC": 2, "ORG": 3, "EVENT": 4, "MISC": 5}

    ## Get nlp and sentiment models
    nlp = stanza.Pipeline(lang='de', processors='tokenize,ner,pos', use_gpu=True)

    with open("analysis_text/wikidata_ids/organization.json", "r") as f:
        org_list = json.load(f)

    with open("analysis_text/wikidata_ids/occurrence_event.json", "r") as f:
        event_list = json.load(f)

    for vi, video in enumerate(args.videos):
        print(f"Processing video [{vi+1}/{len(args.videos)}]: {video}")
        
        video_path = Path(video)

        output_dir = Path(args.pkl_dir) / video_path.stem
        output_file_ner = output_dir / "whisperx_ner.pkl"
        output_file_pos = output_dir / "whisperx_pos.pkl"

        if os.path.exists(output_file_ner) and not args.rewrite:
            print("Already processed. Skipping...")
            continue

        whisperx_file_path = output_dir / "asr_whisperx.pkl"

        whisperx_spk_turns = pickle.load(open(whisperx_file_path, "rb"))["output_data"]["speaker_turns"]

        pos_feat_dict2 = perform_pos(video_path, whisperx_spk_turns, nlp, pos_dict)
        
        print("Number of PoS features:", len(pos_feat_dict2["output_data"]["speakerturn_wise"]), ", Number of speaker turns:", len(whisperx_spk_turns))
        assert len(pos_feat_dict2["output_data"]["speakerturn_wise"]) == len(whisperx_spk_turns)

        ner_feat_dict2 = perform_ner(video_path, whisperx_spk_turns, nlp, ner_dict, org_list, event_list)
        
        print("Number of NER features:", len(ner_feat_dict2["output_data"]["speakerturn_wise"]), ", Number of speaker turns:", len(whisperx_spk_turns))
        assert len(ner_feat_dict2["output_data"]["speakerturn_wise"]) == len(whisperx_spk_turns)
        
        print("Saving to", output_file_pos)
        with open(output_file_pos, "wb") as output_file:
            pickle.dump(pos_feat_dict2, output_file)

        print("Saving to", output_file_ner)
        with open(output_file_ner, "wb") as output_file:
            pickle.dump(ner_feat_dict2, output_file)


if __name__ == "__main__":
    main()
