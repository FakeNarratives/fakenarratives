import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead
from text_utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates image captions for frames sampled from videos using BLIP2 model.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-m", "--model", type=str, default="mpnet", help="Sentence transformer model to use | mpnet | minilm")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for input transcriptions")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def get_model(device, mtype):
    # paraphrase-multilingual-MiniLM-L12-v2 | paraphrase-multilingual-mpnet-base-v2
    if mtype == "minilm":
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    elif mtype == "mpnet":
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    else:
        raise ValueError("Invalid model type")
    return model


def read_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def compute_average_embedding(text_chunks, model):
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    average_embedding = sum(embeddings) / len(embeddings)
    return average_embedding


def break_text(text, chunk_size=128):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i+chunk_size]))

    return chunks

def process_transcript(video_path, asr_dict, model):
    """
    Takes a video asr transcript and compute sentence embeddings

    Args:
        video_path (str): path to video
        asr_dict (dict): dictionary containing asr transcript
        model (SentenceTransformer): sentence transformer model

    Returns:
        video_feat_dict (dict): dictionary containing frame captions

    """

    video_feat_dict = {"github_repo": "https://huggingface.co/sentence-transformers",
                        "commit_id": "ef15aed8b328d308d7237b9bf15269f2cd19e268",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {}
                        }


    asr_output = asr_dict["output_data"]
    speaker_segments = asr_output["speaker_segments"]
    
    # Compute encoding for each speaker segment
    segment_texts = [segment["text"] for segment in speaker_segments]
    segment_encodings = model.encode(segment_texts, convert_to_tensor=True)
    segment_enc_list = []
    for segment, encoding in zip(speaker_segments, segment_encodings):
        segment_enc_list.append({"start": segment["start"], "end": segment["end"], "embedding": encoding.cpu().numpy()})

    # Compute encoding for each speaker-turn by averaging text chunks of length 128
    speaker_turns = get_speaker_turns(speaker_segments)
    turn_enc_list = []
    for turn in speaker_turns:
        if len(turn["text"].split()) <= 128:
            encoding = model.encode(turn["text"], convert_to_tensor=True)
        else:
            text_chunks = break_text(turn["text"])
            encoding = compute_average_embedding(text_chunks, model)

        turn_enc_list.append({"start": turn["start"], "end": turn["end"], "embedding": encoding.cpu().numpy()})

    video_feat_dict["output_data"]["segment_features"] = segment_enc_list
    video_feat_dict["output_data"]["turn_features"] = turn_enc_list

    return video_feat_dict



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(device, args.model)

    ## TODO: Use T5 model to summarize speaker turn text and generate embeddings from it
    # tokenizer = AutoTokenizer.from_pretrained("t5-3b")
    # model = AutoModelWithLMHead.from_pretrained("t5-3b", load_in_8bit=True, device_map="auto")
    # print(model.config.max_position_embeddings)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path)

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "asr_text_features_%s.pkl"%(args.model))):
            asr_dict = pickle.load(open(os.path.join(input_path, "asr.pkl"), "rb"))
            video_path = input_path.replace("results_pkl", "videos")

            video_feat_dict = process_transcript(video_path, asr_dict, model)

            with open(os.path.join(out_loc, "asr_text_features_%s.pkl"%(args.model)), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)
            


if __name__ == "__main__":
    main()
