import os
import sys
import pickle
import argparse
import logging
import torch
import re
from tqdm import tqdm
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Classification with Qwen2VL at 1 FPS")
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-f", "--fps", type=int, default=1, help="fps to process video")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="number of workers to use for data loading")
    parser.add_argument("-r", "--rewrite", action="store_true", help="rewrite existing pickle files")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=960,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


def get_qwenvl(device="cuda"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            # device_map="auto",
        )
    model = model.to(device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    return model, processor


def get_prompt(task):
    if task == "social_roles":
        question = "What is the job or role of the person in this image?"  ## Best prompt 1
        label_map = {
            "A. a celebrity": "A", "B. a doctor": "B", "C. a domain expert": "C", "D. a firefighter": "D", "E. a journalist": "E", 
            "F. a lab worker": "F", "G. a layperson": "G", "H. a medical expert": "H", "I. a military personnel": "I", 
            "J. a news anchor": "J", "K. a news reporter": "K", "L. a nurse": "L",  "M. a politician": "M", "N. a public service worker": "N", 
            "O. a soldier": "O", "P. a testcenter worker": "P", "Q. an NGO worker": "Q", "R. an athlete": "R", "S. an elite": "S", 
            "T. from police": "T", "U. from press or media": "U", "V. No person visible": "V", "W. None of the above": "W"
        }
        coarse_map = {
            "J": "anchor", "K": "reporter", "E": "journalist", "U": "media personnel", "H": "elite", "C": "elite", "R": "elite", 
            "A": "elite", "M": "politician", "I": "elite", "G": "layperson", "B": "doctor", "L": "medical personnel", 
            "F": "medical personnel", "P": "medical personnel", "D": "public safety personnel", "O": "public safety personnel", 
            "T": "public safety personnel", "Q": "public service personnel", "N": "public service personnel", "S": "elite", 
            "V": "No person visible", "W": "None"
        }
        labels_of_interest = ["anchor", "reporter", "journalist", "media personnel", "elite", "politician", "layperson", "doctor",
                              "medical personnel", "public safety personnel", "public service personnel", "No person visible"]
    elif task == "locations":
        question = "What is the exact setting of the place in this image?"  ## Best prompt 5
        label_map = {
            "A. an aerial view or location": "A", "B. an industrial outdoor location": "B", "C. an outdoor location": "C", 
            "D. an outer space location": "D", "E. inside a camp": "E", "F. inside a doctor's office": "F", "G. inside a home": "G", 
            "H. inside a hospital": "H", "I. inside a military building": "I", "J. inside a museum": "J", "K. inside a political building": "K", 
            "L. inside a public indoor location": "L", "M. inside a religious building": "M", "N. inside a restaurant": "N", "O. inside a school": "O", 
            "P. inside a shop": "P", "Q. inside a studio": "Q", "R. inside a test center": "R", "S. inside a train or bus station": "S", 
            "T. inside a vaccination center": "T", "U. inside an indoor location": "U", "V. inside an industrial indoor location": "V",
            "W. inside an office": "W", "X. None of the above": "X"
        }
        coarse_map = {  
            "A": "aerial_outdoor", "B": "industrial_outdoor", "C": "outdoor", "D": "outdoor", "E": "private_indoor", "F": "private_indoor", "G": "private_indoor",
            "H": "public_indoor", "I": "elite_indoor", "J": "public_indoor", "K": "elite_indoor", "L": "public_indoor", "M": "elite_indoor", "N": "public_indoor", 
            "O": "public_indoor", "P": "public_indoor", "Q": "studio", "R": "private_indoor", "S": "public_indoor", "T": "public_indoor", "U": "public_indoor", "V": 
            "industrial_indoor", "W": "private_indoor", "X": "None"
        }
        labels_of_interest = ["aerial", "industrial", "private", "public", "elite", "studio", "indoor", "outdoor"]
    elif task == "events":
        question = "What is the main event shown in the image?"  ## Best prompt 3
        label_map = {
            "A. conference": "A", "B. crime-related fire": "B", "C. cultural ceremony": "C", "D. cultural festival": "D", "E. demonstration": "E", 
            "F. explosion": "F", "G. hostage taking": "G", "H. migration": "H", "I. military action": "I", "J. natural event": "J", "K. religious ceremony": "K", 
            "L. speech": "L", "M. sport activity": "M", "N. voting": "N", "O. war-related fire": "O", "P. None of the above": "P"
            }
        coarse_map = {
            "A": "conference_political", "L": "speech_political", "E": "demonstration_political", "N": "voting_political", "H": "migration_political",
            "C": "ceremony_cultural", "D": "festival_cultural", "K": "ceremony_religious", "M": "sport", "I": "military",
            "O": "fire_war", "F": "explosion_war", "B": "fire_crime", "G": "hostage_crime", "J": "natural", "P": "None"
        }
        labels_of_interest = ["conference", "speech", "demonstration", "voting", "migration", "ceremony", "festival", "sport", "military", "fire", 
                              "explosion", "hostage", "natural", "political", "cultural", "religious", "war", "crime"]

    query = f"{question}\n" + "\n".join(label_map.keys()) + "\nPlease answer only with the option's letter from the given choices directly without any explanation."

    reverse_label_map = {v: k[3:] for k, v in label_map.items()}

    assert len(label_map) == len(coarse_map)

    return query, label_map, reverse_label_map, coarse_map, labels_of_interest


def process_response(response, kwargs):
    response = response.rstrip(".").strip()

    # Process the response
    if len(response) > 1:
        # Check for "answer is X" pattern anywhere in the text
        answer_is_match = re.search(r'answer is ([A-Z])', response, re.IGNORECASE)
        if answer_is_match:
            response = answer_is_match.group(1).upper()
        else:
            # Check if the response matches any key in the label map
            matching_keys = [key for key in kwargs["label_map"].keys() if key in response]
            if matching_keys:
                response = kwargs["label_map"][matching_keys[0]]
            else:
                response = 'Z'
    elif len(response) == 0:
        response = 'Z'

    original_label = kwargs["reverse_label_map"].get(response, 'None')
    response = kwargs["coarse_map"].get(response, 'None')
    res_labels = response.split("_")
    vector = np.array([1 if label in res_labels else 0 for label in kwargs["labels"]])

    return vector, original_label


def process_batch(frames, model, processor, prompt, device="cuda"):
    messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ] * len(frames)

    texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

    inputs = processor(
            text=texts, images=[Image.fromarray(frame) for frame in frames], padding=True, return_tensors="pt"
        ).to(torch.float16).to(device)

    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=10)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts


def process_video(video_path, model, processor, kwargs):
    """
    Take QwenVL2 model and do social roles, locations or event classifcation on the video at a specific FPS.

    Args:
        video_path (str): path to video
        model (Qwen2VLForConditionalGeneration): QwenVL model
        processor (AutoProcessor): QwenVL processor
        kwargs (dict): dictionary containing device, prompt, label_map, reverse_label_map, coarse_map, args

    Returns:
        video_feat_dict (dict): dictionary containing predicted shot scale and movement for each shot in the video.
        {
            "github_repo": "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct",
            "commit_id": "51c47430f97dd7c74aa1fa6825e68a813478097f",
            "parameters": "default",
            "video_file": video_path,
            "output_data": {
                "labels": <labels>,
                "responses": <response vectors>, # Size: (num_labels, num_frames)
                "original_responses": <original responses>, # Size: (num_frames,)
                "times": <time stamps>,
                "delta_time": 1/args.fps,
            }
        }
    """

    video_feat_dict = {"github_repo": "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct",
                        "commit_id": "51c47430f97dd7c74aa1fa6825e68a813478097f",
                        "parameters": f"fps: {kwargs['args'].fps}, max_dimension: {kwargs['args'].max_dimension}",
                        "video_file": video_path,
                        "output_data": {}
                    }

    # Load video
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, kwargs["args"], kwargs["args"].num_workers, kwargs["args"].fps)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

    response_vectors = []
    response_labels = []
    times = []
    batch_frames = []
    for f, frame in enumerate(tqdm(vr)):
        batch_frames.append(frame)
        if len(batch_frames) == kwargs["args"].batch_size:
            responses = process_batch(batch_frames, model, processor, kwargs["prompt"], kwargs["device"])
            vectors, org_res = zip(*[process_response(res, kwargs) for res in responses])
            response_vectors.extend(vectors)
            response_labels.extend(org_res)
            batch_frames = []
        times.append(f/fps)

    if len(batch_frames) > 0:
        responses = process_batch(batch_frames, model, processor, kwargs["prompt"], kwargs["device"])
        vectors, org_res = zip(*[process_response(res, kwargs) for res in responses])
        response_labels.extend(org_res)
        response_vectors.extend(vectors)

    response_vectors = np.array(response_vectors).T

    video_feat_dict["output_data"]["labels"] = kwargs["labels"]
    video_feat_dict["output_data"]["responses"] = response_vectors
    video_feat_dict["output_data"]["original_responses"] = response_labels
    video_feat_dict["output_data"]["times"] = times
    video_feat_dict["output_data"]["delta_time"] = 1/kwargs["args"].fps

    return video_feat_dict



def main():
    args = parse_args()

    set_seeds(42)

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_qwenvl(device)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        for task in ["social_roles", "locations", "events"]:
            if os.path.exists(os.path.join(output_dir, "vlm_%s.pkl"%(task))) and not args.rewrite:
                logging.info(f"\tFound existing pickle file in {output_dir}, skipping")
                continue

            prompt, label_map, reverse_label_map, coarse_map, labels = get_prompt(task)
            kwargs = {
                "device": device,
                "prompt": prompt,
                "label_map": label_map,
                "reverse_label_map": reverse_label_map,
                "coarse_map": coarse_map,
                "labels": labels,
                "args": args
            }

            video_feat_dict = process_video(video_path, model, processor, kwargs)

            with open(os.path.join(output_dir, "vlm_%s.pkl"%(task)), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

            print("%%\n")


if __name__ == "__main__":
    main()