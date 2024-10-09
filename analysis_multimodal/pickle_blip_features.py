import os
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import decord
from decord import VideoReader, cpu
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Model
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipModel, AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generates image captions for frames sampled from videos using BLIP2 model.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size for inference")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def get_model(device):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-flan-t5-xl")
    return processor, tokenizer, model


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:# print(image_features.shape, text_features.shape)
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def process_video(video_path, caption_path, processor, 
                    tokenizer, model, 
                    batch_size=128, device="cuda"):
    """
    Takes the blip2 frame captions and extracts image and text features for each frame and that caption respectively.

    Args:
        video_path (str): path to video
        caption_dict (dict): dictionary containing frame index and corresponding caption
        tokenizer (AutoTokenizer): tokenizer for BLIP2 model
        model (Blip2Model): BLIP2 model
        batch_size (int, optional): batch size for inference. Defaults to 128.
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary frame features and caption features with mean, max and last token representations
    """

    video_feat_dict = {"github_repo": "https://github.com/salesforce/LAVIS",
                        "commit_id": "baad2d7c8df599d8d9b081ba2e946626eaa2dc34",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {}
                        }

    blip_img_features, blip_text_mean, blip_text_max, blip_text_last = {}, {}, {}, {}

    videoreader = VideoReader(video_path+".mp4", num_threads=1, ctx=cpu(0))

    caption_dict = pickle.load(open(os.path.join(caption_path, "blip_captions.pkl"), "rb"))["output_data"]

    frame_list = list(caption_dict.keys())

    ## Do batched inference over frame_list and combine the results indexed by frame_list
    for i in tqdm(range(0, len(frame_list), batch_size)):
        batch_frame_list = frame_list[i:i+batch_size]
        batch_frames = videoreader.get_batch(batch_frame_list)
        batch_captions = [caption_dict[frame_idx] for frame_idx in batch_frame_list]

        with torch.no_grad():
            image_inputs = processor(batch_frames, return_tensors="pt").to(device, torch.float16)
            image_features = model.get_image_features(**image_inputs, return_dict=True)["pooler_output"]

            text_inputs = tokenizer(batch_captions, padding=True, return_tensors="pt").to(device)

            ## Size of last_hidden_state: (batch_size, seq_len, 2048) 
            last_hidden_state = model.get_text_features(**text_inputs, decoder_input_ids=text_inputs.input_ids, return_dict=True)["encoder_last_hidden_state"]
            
            # Average Pooling
            avg_pooling = torch.mean(last_hidden_state, dim=1).cpu().numpy()

            # Max Pooling
            max_pooling = torch.max(last_hidden_state, dim=1)[0].cpu().numpy()

            # Last Token representation
            last_token = last_hidden_state[:, -1, :].cpu().numpy()

        for j, frame_idx in enumerate(batch_frame_list):
            blip_img_features[frame_idx] = image_features[j]
            blip_text_mean[frame_idx] = avg_pooling[j]
            blip_text_max[frame_idx] = max_pooling[j]
            blip_text_last[frame_idx] = last_token[j]

    video_feat_dict["output_data"]["image"] = blip_img_features
    video_feat_dict["output_data"]["caption_mean"] = blip_text_mean
    video_feat_dict["output_data"]["caption_max"] = blip_text_max
    video_feat_dict["output_data"]["caption_last"] = blip_text_last

    return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    processor, tokenizer, model = get_model(device)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print(f"Video {i+1}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "blip_imcap_features.pkl")):
            video_feat_dict = process_video(input_path, out_loc, processor, tokenizer, model, args.batch_size, device)

            with open(os.path.join(out_loc, "blip_imcap_features.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)


if __name__ == "__main__":
    main()
