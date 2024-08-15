import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from audio_utils import *

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def sort_tracks_by_start_time(asd_tracks):
    return sorted(asd_tracks, key=lambda x: x['frames'][0])

def find_relevant_track(turn, sorted_asd_tracks, fps):
    turn_start, turn_end = turn['start'], turn['end']
    relevant_tracks = []
    for track in sorted_asd_tracks:
        track_start = track['frames'][0] / fps
        track_end = track['frames'][-1] / fps
        if track_start >= turn_end:
            break  # No need to check further tracks
        if track_end > turn_start and track_start < turn_end:
            overlap_start = max(turn_start, track_start)
            overlap_end = min(turn_end, track_end)
            overlap_duration = overlap_end - overlap_start
            relevant_tracks.append((track, overlap_duration))
    
    if not relevant_tracks:
        return None
    
    # Choose the track with the highest overlap duration
    return max(relevant_tracks, key=lambda x: x[1])[0]

def calculate_overlap(turn, track, fps):
    turn_start, turn_end = turn['start'], turn['end']
    track_start = track['frames'][0] / fps
    track_end = track['frames'][-1] / fps
    overlap_start = max(turn_start, track_start)
    overlap_end = min(turn_end, track_end)
    overlap_duration = max(0, overlap_end - overlap_start)
    turn_duration = turn_end - turn_start
    print(turn_duration)
    return overlap_duration / turn_duration if turn_duration > 0 else 0


def update_speaker_turns(args):
    for video_path in args.videos:
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        # Load face detection
        face_detection = load_pickle(os.path.join(output_dir, "face_detection_insightface.pkl"))
        fps = face_detection['args']['fps']

        # Load ASR results
        asr_path = os.path.join(output_dir, "asr_whisperx.pkl")
        asr_data = load_pickle(asr_path)
        speaker_turns = get_speaker_turns(asr_data['output_data']['speaker_segments'])

        # Load ASD results
        asd_path = os.path.join(output_dir, "asd_light-asd.pkl")
        asd_data = load_pickle(asd_path)

        # Sort ASD tracks by start time
        sorted_asd_tracks = sort_tracks_by_start_time(asd_data['tracks'])

        # Update speaker turns with ASD information
        for turn in tqdm(speaker_turns, desc=f"Processing {vidname}"):
            relevant_track = find_relevant_track(turn, sorted_asd_tracks, fps)
            if relevant_track:
                overlap_ratio = calculate_overlap(turn, relevant_track, fps)
                turn['active'] = True if overlap_ratio >= args.threshold else False
                turn['active_ratio'] = overlap_ratio
                turn['active_track_id'] = relevant_track['track_id']
            else:
                turn['active'] = False
                turn['active_ratio'] = 0
                turn['active_track_id'] = None

            # print(turn["start"], turn["end"], turn["active"], turn["active_ratio"], turn["active_track_id"])
            # print(turn["start"]/60, turn["end"]/60, turn["active"], turn["active_ratio"], turn["active_track_id"])
            

        # # Save updated ASR results
        # save_pickle(asr_data, asr_path)
        # print(f"Updated ASR results saved to {asr_path}")
        break


def parse_args():
    parser = argparse.ArgumentParser(description="Combine speaker turns with ASD results")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the directory containing pkl files")
    parser.add_argument("--threshold", type=float, default=0.7, help="threshold for determining active speaker turn")
    return parser.parse_args()

def main():
    args = parse_args()
    update_speaker_turns(args)

if __name__ == "__main__":
    main()