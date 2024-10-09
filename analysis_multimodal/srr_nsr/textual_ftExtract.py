import os
import pickle
import stanza
import yaml
import json
import torch
import random
import numpy as np
from germansentiment import SentimentModel
from sentence_transformers import SentenceTransformer

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True
    torch.set_grad_enabled(False)

def get_speaker_turns(speaker_segments, gap=0.01):
    speaker_segments = sorted(speaker_segments, key=lambda x: x["start"])

    speaker_turns = []

    for segment in speaker_segments:

        segment["n_words"] = len(segment["text"].split()) if "text" in segment else 0

        if "speaker" not in segment or segment["speaker"] is None:
            segment["speaker"] = "Unknown"

        if not speaker_turns:
            seg = {"start_time": segment["start"], "end_time": segment["end"], 
                   "text": segment["text"], "speaker": segment["speaker"], "n_words": segment["n_words"]}
            speaker_turns.append(seg)
        else:
            last_turn = speaker_turns[-1]

            ## Checking if current segment belongs to same speaker
            if last_turn["speaker"] == segment["speaker"]:
                # last_turn["end"] = max(last_turn["end"], segment["end"]) # If no gap to be add
                last_turn["end_time"] = max(last_turn["end_time"], segment["end"] - gap)   # Just to ensure some gap between turns

                last_turn["text"] += " " + segment["text"].strip()
                last_turn["n_words"] += len(segment["text"].split())
            else:
            ## Treat otherwise as a new speaker turn
                if segment["start"] - last_turn["end_time"] < gap:       ## Comment if no gap to add
                    segment["start"] = last_turn["end_time"] + gap
                
                seg = {"start_time": segment["start"], "end_time": segment["end"], 
                   "text": segment["text"], "speaker": segment["speaker"], "n_words": segment["n_words"]}
                
                speaker_turns.append(seg)
    
    return speaker_turns


def sentiment_analysis(video_path, output_dir, model):
    """
    Predict sentiments based on ASR transcript
    github: https://huggingface.co/mdraw/german-news-sentiment-bert
    """

    vidname = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{output_dir}/{vidname}", exist_ok=True)
    sentiment_pkl = f"{output_dir}/{vidname}/icmr_sentiment.pkl"
    asr_pkl = f"{output_dir}/{vidname}/asr_whisperx.pkl"

    # if os.path.exists(sentiment_pkl):
    #     print(f"\t[Sentiment] Found pkl: {sentiment_pkl} , skip Sentiment Analysis", flush=True)
    #     return
    if not os.path.exists(asr_pkl):
        print("\t[Sentiment] Please provide a pkl containing an ASR transcript (--asr) before approaching Sentiment Analysis", flush=True)
        return
    
    with open(asr_pkl, 'rb') as pkl:
        asr_data = pickle.load(pkl)["output_data"]
    
    # store all sentences in one array
    texts = []
    for segment in asr_data['speaker_segments']:
        texts.append(segment['text'].strip())

    preds, probs = model.predict_sentiment(texts, output_probabilities=True)
    
    sentiments = []
    for i, segment in enumerate(asr_data['speaker_segments']):
        segment_data = {}
        segment_data['start'] = segment['start']
        segment_data['end'] = segment['end']
        segment_data['text'] = segment['text'].strip()
        segment_data['sentiment'] = preds[i]
        segment_data['sentiment_probs'] = probs[i]
        sentiments.append(segment_data)
    
    # store in pkl
    with open(sentiment_pkl, 'wb') as pkl:
        pickle.dump(sentiments, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\t[Sentiment] Successfully applied Sentiment Analysis! Result: {pkl.name}', flush=True)


def sentence_embeddings(video_path, output_dir, stanza_nlp, model):
    """
    Create text embeddings of every frame using CLIPs image encoder.
    These are later used to calculate similarities between frames.
    """
    vidname = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{output_dir}/{vidname}", exist_ok=True)
    embeddings_pkl = f"{output_dir}/{vidname}/sentence_embeddings.pkl"
    asr_pkl = f"{output_dir}/{vidname}/asr_whisperx.pkl"

    # if os.path.exists(embeddings_pkl):
    #     print(f"\t[Sentence Embeddings] Found pkl: {embeddings_pkl} , skip Sentence Encoding", flush=True)
    #     return
    if not os.path.exists(asr_pkl):
        print("\t[Sentence Embeddings] Please provide a pkl containing Speaker Diariazation data (--diarize) before approaching Sentence Encoding", flush=True)
        return
    
    with open(asr_pkl, 'rb') as pkl:
        asr_data = pickle.load(pkl)["output_data"]

    speaker_turns = asr_data['speaker_turns']

    embeddings = []
    for segment in speaker_turns:
        text = stanza_nlp(segment['text'])
        # get sentences from text segment
        sentences = [sent.text for sent in text.sentences]
        # convert sentences to their embeddigns (return a list where each entry is the embedding of a sentence)
        sentence_embeddings = model.encode(sentences)
        embeddings.append({"start_time": segment['start'], "end_time": segment['end'], "sentences": sentences, "sentence_embeddings": sentence_embeddings})
    
    # store result as pkl
    with open(embeddings_pkl, 'wb') as pkl:
        pickle.dump(embeddings, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[Sentence Embeddings] Successfully extracted Sentence Embeddings! Result: {pkl.name}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract visual features from videos')
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help='path to the output folder')
    args = parser.parse_args()

    set_seeds(42)

    ## Load models
    stanza_nlp = stanza.Pipeline(lang='de', processors='tokenize,ner,pos', download_method=None, logging_level='ERROR')
    sent_model = SentimentModel('mdraw/german-news-sentiment-bert')
    emb_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    videos = args.videos
    for vi, video_path in enumerate(videos):
        print(f"Processing video {vi+1}/{len(videos)}: {video_path}", flush=True)
        
        # Sentiment Analysis
        print("\t[Sentiment]", flush=True)
        sentiment_analysis(video_path, args.pkl_dir, sent_model)

        # Sentence Embeddings
        print("\t[Sentence Embeddings]", flush=True)
        sentence_embeddings(video_path, args.pkl_dir, stanza_nlp, emb_model)

        print()

if __name__ == "__main__":
    main()