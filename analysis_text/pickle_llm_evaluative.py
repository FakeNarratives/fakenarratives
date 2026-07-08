import os
import pickle
import re
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from text_utils import *
from pathlib import Path
sys.path.append(".")
from project_utils import set_seeds, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Runs LLM inference on ASR Speaker Turns")
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


def get_model():
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    return model, tokenizer


def perform_llm_eval(video_path, speaker_turns, model, 
                     tokenizer, alpaca_prompt, prompt_type, device):
    """
    Takes ASR Speaker Turns and runs LLM inference on them

    Args:
        video_path: str Path to video file
        speaker_turns: list List of speaker turns
        model: FastLanguageModel Model object
        tokenizer: Tokenizer object
        alpaca_prompt: str Alpaca prompt
        prompt_type: str Prompt type

    Returns:
        dict Dictionary containing sentiment features

    """

    video_feat_dict = {"github_repo": "https://github.com/unslothai/unsloth",
                        "commit_id": "38663b01f5dd0e610b12475bd95b144303cff539",
                        "parameters": "default",
                        "video_file":  video_path,
                        "output_data": []
                    }


    llm_outputs = []
        
    for segment in tqdm(speaker_turns):
        if segment["end"] <= segment["start"]:
            llm_outputs.append({"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"],
                                        "label": "None", "confidence": "None"})

            continue
        
        text = segment["text"].strip()
        
        input = tokenizer(
        [
            alpaca_prompt.format(
                prompt_type, # instruction
                text, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors = "pt").to(device)

        output = model.generate(**input, max_new_tokens = 128, temperature=0.7, top_p=0.8, 
                                top_k=20, repetition_penalty=1.05, use_cache = True)
        
        response = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split("Response:")[1].strip()
        
        seg_output = {"start": segment["start"], "end": segment["end"], "speaker": segment["speaker"]}
        if "\"label\": \"evaluative\"" in response:
            seg_output["label"] = "evaluative"
        elif "\"label\": \"non-evaluative\"" in response:
            seg_output["label"] = "non-evaluative"
        else:
            seg_output["label"] = "non-evaluative"
        
        if "\"confidence\": \"high\"" in response:
            seg_output["confidence"] = "high"
        elif "\"confidence\": \"moderate\"" in response:
            seg_output["confidence"] = "moderate"
        elif "\"confidence\": \"low\"" in response:
            seg_output["confidence"] = "low"
        else:
            seg_output["confidence"] = "low"
        
        llm_outputs.append(seg_output)

    video_feat_dict["output_data"] = llm_outputs

    return video_feat_dict



def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    set_seeds(42)

    model, tokenizer = get_model()

    ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                    ### Instruction:
                    {}

                    ### Input:
                    {}

                    ### Response:
                    {}"""
    
    TASK_PROMPT = """
            Your task is to detect the presence or absence of evaluative language in German input text.
            Evaluative language can involve positive or negative judgments, comparisons (better, worse), or expressions of subjective perceptions (apparently, obviously).

            Output directly in the following JSON format without any additional information or explanation:
                {
                    "confidence": "low, moderate, high",    ## Confidence in the decision
                    "label": "evaluative or non-evaluative"    ## Final label
                }
            """

    for vi, video in enumerate(args.videos):
        print(f"Processing video [{vi+1}/{len(args.videos)}]: {video}")
        
        video_path = Path(video)

        output_dir = Path(args.pkl_dir) / video_path.stem
        output_file = output_dir / "llm_evaluative.pkl"

        if os.path.exists(output_file) and not args.rewrite:
            print("Already processed. Skipping...")
            continue

        pickle_file_path = output_dir / "asr_whisperx.pkl"

        try:
            whisperx_spk_turns = pickle.load(open(pickle_file_path, "rb"))["output_data"]["speaker_turns"]
        except FileNotFoundError:
            print("ASR file not found. Skipping...")
            continue

        llm_feat_dict = perform_llm_eval(video_path, whisperx_spk_turns, model, 
                                         tokenizer, ALPACA_PROMPT, TASK_PROMPT,
                                         device)
        
        print("Number of LLM features:", len(llm_feat_dict["output_data"]), ", Number of speaker turns:", len(whisperx_spk_turns))
        assert len(llm_feat_dict["output_data"]) == len(whisperx_spk_turns)
        print("Saving to", output_file)

        with open(output_file, "wb") as output_file:
            pickle.dump(llm_feat_dict, output_file)


if __name__ == "__main__":
    main()
