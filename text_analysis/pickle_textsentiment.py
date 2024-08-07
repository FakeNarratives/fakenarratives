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

    return [os.path.join(base_input_dir, re.sub(r'\.([^.]+)$', '', vp)) for vp in transcript_paths], \
                [os.path.join(base_output_dir, re.sub(r'\.([^.]+)$', '', vp)) for vp in transcript_paths]


def get_sentiment_model(type="news"):
    if type == "news":
        sen_model = SentimentModel('mdraw/german-news-sentiment-bert')
    elif type == "general":
        sen_model = SentimentModel()
    elif type == "multi":
        sen_model = pipeline(
                model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
                top_k=None
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
                        "video_file":  video_path.replace("results_pkl", "videos"),
                        "output_data": {
                            "sentence_wise": {},
                            "speakerturn_wise": {},
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

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_transcript_paths(args.file, args.inp_dir, args.out_dir)

    nlp = stanza.Pipeline(lang='de', processors='tokenize', use_gpu=True)

    ## Sentiment models
    models = {
        "news": get_sentiment_model("news"),
        "general": get_sentiment_model("general"),
        "multi": get_sentiment_model("multi")
    }

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video [{i+1}/{len(input_paths)}]\t{input_path}")
        
        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if os.path.exists(os.path.join(out_loc, "whisperx_sentiment.pkl")) and not args.rewrite:
            print("Already processed. Skipping...")
            continue

        whisperx_segments = pickle.load(open(os.path.join(input_path, "asr_whisperx.pkl"), "rb"))["output_data"]["speaker_segments"]
        
        speaker_turns = get_speaker_turns(whisperx_segments)

        sentiment_feat_dict = perform_sentiment(input_path, speaker_turns, nlp, models, sent_dict)
        
        print("Number of news model sentiment features:", len(sentiment_feat_dict["output_data"]["model_news"]["speakerturn_wise"]), ", Number of speaker turns:", len(speaker_turns))
        print("Number of general model sentiment features:", len(sentiment_feat_dict["output_data"]["model_general"]["speakerturn_wise"]), ", Number of speaker turns:", len(speaker_turns))
        print("Number of multi model sentiment features:", len(sentiment_feat_dict["output_data"]["model_multi"]["speakerturn_wise"]), ", Number of speaker turns:", len(speaker_turns))
        assert len(sentiment_feat_dict["output_data"]["model_news"]["speakerturn_wise"]) == len(speaker_turns)
        assert len(sentiment_feat_dict["output_data"]["model_general"]["speakerturn_wise"]) == len(speaker_turns)
        assert len(sentiment_feat_dict["output_data"]["model_multi"]["speakerturn_wise"]) == len(speaker_turns)

        print("Saving to", os.path.join(out_loc, "whisperx_sentiment.pkl"))
        with open(os.path.join(out_loc, "whisperx_sentiment.pkl"), "wb") as output_file:
            pickle.dump(sentiment_feat_dict, output_file)


if __name__ == "__main__":
    main()
