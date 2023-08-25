import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import decord
from decord import VideoReader, cpu
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Model
# from transformers import BlipProcessor, BlipForConditionalGeneration, BlipModel, AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates image captions for frames sampled from videos using BLIP2 model.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size for inference | 10GB GPU with 128 batch size")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def get_model(device):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")
    return processor, model


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def process_video(video_path, processor, model,
                    batch_size=128, device="cuda"):
    """
    Takes a video and generates captions for frames sampled at 1 fps.
    Extracts image features for each frame and text features for each caption.

    Args:
        video_path (str): path to video
        processor (Blip2Processor): BLIP2 processor
        model (BlipForConditionalGeneration): BLIP2 model
        batch_size (int, optional): batch size for inference. Defaults to 128.
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing frame captions

    """

    video_feat_dict = {"github_repo": "https://github.com/salesforce/LAVIS",
                        "commit_id": "baad2d7c8df599d8d9b081ba2e946626eaa2dc34",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {}
                        }

    videoreader = VideoReader(video_path+".mp4", num_threads=1, ctx=cpu(0))

    fps_t = 1
    total_frames = len(videoreader)
    frame_step = int(videoreader.get_avg_fps() // fps_t)

    print(f"Total frames: {total_frames}, frame step: {frame_step}")

    frame_list = list(range(0, total_frames, frame_step))

    ## Do batched inference over frame_list and combine the results indexed by frame_list
    for i in tqdm(range(0, len(frame_list), batch_size)):
        batch_frame_list = frame_list[i:i+batch_size]
        batch_frames = videoreader.get_batch(batch_frame_list)

        with torch.no_grad():
            input = processor(batch_frames, return_tensors="pt").to(device, torch.float16)

            generated_ids = model.generate(**input, max_new_tokens=30)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for j, frame_idx in enumerate(batch_frame_list):
            video_feat_dict["output_data"][frame_idx] = generated_texts[j].strip()

    return video_feat_dict



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    processor, model = get_model(device)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print(f"Video {i+1}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "blip_captions.pkl")):
            video_feat_dict = process_video(input_path, processor, model, args.batch_size, device)

            with open(os.path.join(out_loc, "blip_captions.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)
            


if __name__ == "__main__":
    main()
