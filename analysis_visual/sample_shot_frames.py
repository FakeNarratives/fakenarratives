import os
import pickle
import argparse
from decord import VideoReader, cpu
from PIL import Image
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Samples a frame per shot from a video.")
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Base output directory for frames")
    parser.add_argument("-s", "--shots", type=str, required=True, help="Path to the transnet_shotdetection.pkl file")
    args = parser.parse_args()
    return args

def sample_frame(vr, start_time, end_time, fps):
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Sample one random frame within the shot
    frame_index = random.randint(start_frame, end_frame)
    
    frame = vr.get_batch([frame_index]).asnumpy()[0]
    return frame

def process_video(video_path, output_base_path, shots_path):
    with open(shots_path, 'rb') as f:
        shot_dict = pickle.load(f)
    print("Number of shots: ", len(shot_dict["output_data"]["shots"]))

    vr = VideoReader(video_path, num_threads=12, ctx=cpu(0))
    fps = vr.get_avg_fps()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = video_name.replace(":", "-")
    frames_dir = os.path.join(output_base_path, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    for i, shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        start_time, end_time = shot["start"], shot["end"]
        frame = sample_frame(vr, start_time, end_time, fps)

        frame_img = Image.fromarray(frame)
        frame_filename = f"{video_name}_shot{i:03d}.jpg"
        frame_img.save(os.path.join(frames_dir, frame_filename))

def main():
    args = parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.shots):
        print(f"Error: Shot detection file not found: {args.shots}")
        return
    
    frames_base_dir = os.path.join(args.out_dir)
    if not os.path.exists(frames_base_dir):
        os.makedirs(frames_base_dir)

    print(f"Processing Video: {args.video}")
    process_video(args.video, args.out_dir, args.shots)
    print("Processing complete.")

if __name__ == "__main__":
    main()