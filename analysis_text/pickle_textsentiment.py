import os
import pickle
import re
import numpy as np
import stanza
import torch
from germansentiment import SentimentModel
from transformers import pipeline
from text_utils import *
import argparse
from pathlib import Path
import sys
sys.path.append(".")
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Runs german sentiment detection model on ASR transcript")
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


def get_sentiment_model(type, device):
    if type == "news":
        sen_model = SentimentModel('mdraw/german-news-sentiment-bert')
    elif type == "general":
        sen_model = SentimentModel()
    elif type == "multi":
        sen_model = pipeline(
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                top_k=None,
                device=device
            )
    return sen_model



def extract_sentiment(model, text, model_type):
    if model_type in ["news", "general"]:
        preds, probs = model.predict_sentiment([text], output_probabilities=True)
        return preds[0], [probs[0][0][1], probs[0][1][1], probs[0][2][1]]
    elif model_type == "multi":
        result = model(text, truncation=True, max_length=512)[0]
        pred = max(result, key=lambda x: x['score'])['label']
        probs = [item['score'] for item in result]
        return pred, probs


def perform_sentiment(video_path, speaker_turns, nlp, models, sent_dict):
    """
    Takes ASR segments and performs sentiment analysis of text segments

    Args:
        video_path: str Path to video file
        speaker_turns: list List of speaker turns
        nlp: stanza.Pipeline Stanza NLP pipeline
        models: dict Dictionary of sentiment models
        sent_dict: dict Sentiment label mapping

    Returns:
        dict Dictionary containing sentiment features

    """

    github_map = {"news": "https://huggingface.co/mdraw/german-news-sentiment-bert",
                  "general": "https://huggingface.co/oliverguhr/german-sentiment-bert",
                  "multi": "https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student"}
    
    commit_map = {"news": "6e3e756d9242be48e27f6233a6c1e557726c1bfa",
                    "general": "f195511fd2678d4a56ca3f5a371844138a3bc8d9",
                    "multi": "2e33845d25b3ed0c8994ed53adb72566a1d39d79"}

    video_feat_dict = {"github_repo": ";".join([github_map[m] for m in ["news", "general", "multi"]]),
                        "commit_id": ";".join([commit_map[m] for m in ["news", "general", "multi"]]),
                        "parameters": "default",
                        "video_file":  video_path,
                        "output_data": {
                            "sent_labelmap": sent_dict
                        }
                    }


    for model_type, model in models.items():
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
            
            sent_vector = np.zeros(3)
            for sentence in turn_sentences:
                pred, _ = extract_sentiment(model, sentence, model_type)
                sent_vector[sent_dict[pred]] += 1

            sent_vector = sent_vector / len(turn_sentences)
        
            sentwise_sentiments.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                        "vector": sent_vector})
        
            # Speaker-Turn-wise sentiment
            st_pred, st_prob = extract_sentiment(model, segment["text"].strip(), model_type)
            
            spkturn_sentiments.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                             "pred": st_pred, "prob": st_prob})

        
        video_feat_dict["output_data"][f"model_{model_type}"] = {
            "sentence_wise": sentwise_sentiments,
            "speakerturn_wise": spkturn_sentiments
        }

    return video_feat_dict



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seeds(42)

    sent_dict = {"positive": 0, "negative": 1, "neutral": 2}

    nlp = stanza.Pipeline(lang='de', processors='tokenize', use_gpu=True)

    ## Sentiment models
    models = {
        "news": get_sentiment_model("news", device),
        "general": get_sentiment_model("general", device),
        "multi": get_sentiment_model("multi", device)
    }

    for vi, video in enumerate(args.videos):
        print(f"Processing video [{vi+1}/{len(args.videos)}]: {video}")
        
        video_path = Path(video)

        output_dir = Path(args.pkl_dir) / video_path.stem
        output_file = output_dir / "whisperx_sentiment.pkl"

        if os.path.exists(output_file) and not args.rewrite:
            print("Already processed. Skipping...")
            continue

        whisperx_file_path = output_dir / "asr_whisperx.pkl"

        whisperx_spk_turns = pickle.load(open(whisperx_file_path, "rb"))["output_data"]["speaker_turns"]

        sentiment_feat_dict = perform_sentiment(video_path, whisperx_spk_turns, nlp, models, sent_dict)
        
        print("Number of news model sentiment features:", len(sentiment_feat_dict["output_data"]["model_news"]["speakerturn_wise"]), ", Number of speaker turns:", len(whisperx_spk_turns))
        print("Number of general model sentiment features:", len(sentiment_feat_dict["output_data"]["model_general"]["speakerturn_wise"]), ", Number of speaker turns:", len(whisperx_spk_turns))
        print("Number of multi model sentiment features:", len(sentiment_feat_dict["output_data"]["model_multi"]["speakerturn_wise"]), ", Number of speaker turns:", len(whisperx_spk_turns))
        assert len(sentiment_feat_dict["output_data"]["model_news"]["speakerturn_wise"]) == len(whisperx_spk_turns)
        assert len(sentiment_feat_dict["output_data"]["model_general"]["speakerturn_wise"]) == len(whisperx_spk_turns)
        assert len(sentiment_feat_dict["output_data"]["model_multi"]["speakerturn_wise"]) == len(whisperx_spk_turns)

        print("Saving to", output_file)
        with open(output_file, "wb") as output_file:
            pickle.dump(sentiment_feat_dict, output_file)


if __name__ == "__main__":
    main()
