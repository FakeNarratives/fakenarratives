import os
import pickle
import yaml
import math
import json
import clip
import torch
import random
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import sys
from sklearn.neighbors import KernelDensity
sys.path.append('../../')
from project_utils import read_video_and_get_info

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

def shot_density(video_path, output_dir):
    """
    Calculate shot density from previously saved shot boundary detection result .pkl

    Stores the result in a .pkl file
    """
    # check if shot density was already calculated for the requested file
    vidname = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{output_dir}/{vidname}", exist_ok=True)
    pkl_file = f"{output_dir}/{vidname}/shot_density.pkl"
    if os.path.exists(pkl_file):
        print(f"\t[Shot Density] Found pkl: {pkl_file} , skip Shot Density calculation", flush=True)
        return

    sbd_pkl = f"{output_dir}/{vidname}/transnet_shotdetection.pkl"
    print(sbd_pkl)
    if not os.path.exists(sbd_pkl):
        print("\t[Shot Density] No Shot Boundary Detection .pkl found! Please provide SBD results before applying Shot Density calculation", flush=True)
        return

    with open(sbd_pkl, 'rb') as pkl:
        shots_data = pickle.load(pkl)["output_data"]["shots"]

    last_shot_end = 0
    shots = []
    # shots_data contains tuples in the form of (shot.start, shot.end)
    for shot in shots_data:
        shots.append(shot["start"])

        if shot["end"]> last_shot_end:
            last_shot_end = shot["end"]

    time = np.linspace(0, last_shot_end, math.ceil(last_shot_end * 25.0) + 1)[:, np.newaxis]
    shots = np.asarray(shots).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=10.0).fit(shots)
    log_dens = kde.score_samples(time)
    shot_density = np.exp(log_dens)
    shot_density = (shot_density - shot_density.min()) / (shot_density.max() - shot_density.min())

    output_data = {
        "video_file": video_path,
        "parameters": {"fps": 25.0, "bandwidth": 10.0},
        "output_data": {
                    "y": shot_density.squeeze(),
                    "time": time.squeeze().astype(np.float64),
                    "delta_time": 1 / 25.0
                }
    }
    
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(output_data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[Shot Density] Successfully calculated Shot Density! Result: {pkl.name}", flush=True)



def process_query_batch(batch, text_features, model, preprocessor, device, result_matrices, queries):
    inputs = torch.stack([preprocessor(img) for img in batch]).to(device)
    
    with torch.no_grad(), torch.autocast(device):
        image_features = model.encode_image(inputs)
    
    for domain, domain_queries in queries.items():
        batch_results = []
        for _ in range(len(batch)):
            max_queries = []
            for label, texts in domain_queries.items():
                text_embeds = text_features[domain][label]
                logits_per_image = torch.matmul(image_features, text_embeds.t())
                max_query = max(zip(texts, [label] * len(texts), logits_per_image[_].tolist()), key=lambda item: item[2])
                max_queries.append(max_query)
            
            similarity_scores = torch.tensor([x[2] for x in max_queries], dtype=torch.float32)
            probs = similarity_scores.softmax(dim=-1).cpu().numpy()
            batch_results.append(list(zip(domain_queries.keys(), probs.tolist())))
        
        result_matrices[domain].extend(batch_results)


def predict_CLIP_queries(video_path, output_dir, queries, model, processor, args, device, batch_size=1024):
    vidname = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{output_dir}/{vidname}", exist_ok=True)
    pkl_file = f"{output_dir}/{vidname}/clip_qas.pkl"
    
    if os.path.exists(pkl_file):
        print(f"\t[CLIP] Found pkl: {pkl_file}, skipping", flush=True)
        return

    try:
        video_decoder, _, _, fps, _ = read_video_and_get_info(video_path, args, args.workers, 2)
    except Exception as e:
        print(f"\t[CLIP] Error reading video: {e}", flush=True)
        return
    
    result_matrices = {domain: [] for domain in queries.keys()}

    text_features = {domain: {label: model.encode_text(clip.tokenize(texts).to(device)) 
                              for label, texts in domain_queries.items()}
                     for domain, domain_queries in queries.items()}
    
    batch = []
    for frame in tqdm(video_decoder, desc="Processing frames"):
        batch.append(Image.fromarray(frame))
        
        if len(batch) == batch_size:
            process_query_batch(batch, text_features, model, processor, device, result_matrices, queries)
            batch = []

    if batch:
        process_query_batch(batch, text_features, model, processor, device, result_matrices, queries)

    with open(pkl_file, 'wb') as pkl:
        pickle.dump(result_matrices, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[CLIP] Result saved: {pkl.name}", flush=True)



def process_embedding_batch(batch, model, processor, device):
    inputs = torch.stack([processor(img) for img in batch]).to(device)
    with torch.no_grad(), torch.autocast(device):
        return model.encode_image(inputs).cpu().numpy()


def clip_image_embeddings(video_path, output_dir, model, processor, args, device, batch_size=1024):
    """
    Create image embeddings of every frame using SigLIP's image encoder in batches.
    These are later used to calculate similarities between frames.
    """
    vidname = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(f"{output_dir}/{vidname}", exist_ok=True)
    pkl_file = f"{output_dir}/{vidname}/clip_embeddings.pkl"
    
    if os.path.exists(pkl_file):
        print(f"\t[Image Embeddings] Found pkl: {pkl_file}, skipping", flush=True)
        return

    try:
        video_decoder, _, _, fps, _ = read_video_and_get_info(video_path, args, args.workers, 2)
    except Exception as e:
        print(f"\t[CLIP] Error reading video: {e}", flush=True)
        return
    
    embeddings = []
    batch = []
    for frame in tqdm(video_decoder, desc="Processing frames"):
        batch.append(Image.fromarray(frame))

        if len(batch) == batch_size:
            embeddings.extend(process_embedding_batch(batch, model, processor, device))
            batch = []

    if batch:
        embeddings.extend(process_embedding_batch(batch, model, processor, device))

    with open(pkl_file, 'wb') as pkl:
        pickle.dump(embeddings, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\t[Image Embeddings] Result saved: {pkl.name}", flush=True)



def main():
    parser = argparse.ArgumentParser(description='Extract visual features from videos')
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-f", '--feature', type=str, default='shot-bd', help='Feature to extract (shot-bd, clip)')
    parser.add_argument("-w", '--workers', type=int, default=4, help='Number of workers to use for video decoding')
    parser.add_argument("-o", '--pkl_dir', type=str, required=True, help='path to the output folder')
    args = parser.parse_args()

    set_seeds(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Load model
    if args.feature == 'clip':
        model, processor = clip.load("ViT-B/32", device=device)
        model.eval()
        args.max_dimension = 720

        # clip queries json
        with open("CLIP_queries.json") as f:
            queries = json.load(f)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        print(f"Processing video {vi+1}/{len(videos)}: {video_path}", flush=True)
        if args.feature == "shot-bd":
            print("\t[SHOT DENSITY] Processing Shot Density calculation...", flush=True)
            shot_density(video_path, args.pkl_dir)
        elif args.feature == "clip":
            print("\t[CLIP] Processing CLIP modeling...", flush=True)
            predict_CLIP_queries(video_path, args.pkl_dir, queries, model, processor, args, device)

            print("\t[IMAGE EMBEDDINGS] Processing Image Embeddings extraction...", flush=True)
            clip_image_embeddings(video_path, args.pkl_dir, model, processor, args, device)

        print()

if __name__ == "__main__":
    main()