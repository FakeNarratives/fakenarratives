import pickle
import yaml
import os
import argparse
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_pickle(path):
    with open(path, 'rb') as pkl:
        data = pickle.load(pkl)
    return data

## Processor Functions for each type of feature
## Visual Processing
def process_clip_data(clip_data, segment_start, segment_end, config):
    clip_features = []
    for domain in clip_data.keys():
        relevant_frames = []
        for i, data in enumerate(clip_data[domain]):
            frame_time = i / config['fps']
            if segment_start <= frame_time <= segment_end:
                relevant_frames.append(data)

        label_sums = {}
        for frame in relevant_frames:
            for label, prob in frame:
                if label not in label_sums:
                    label_sums[label] = prob
                else:
                    label_sums[label] += prob

        feature_vec = []
        for label in label_sums.keys():
            avg_prob = label_sums[label] / len(relevant_frames)
            feature_vec.append(avg_prob)

        clip_features.append(feature_vec)
    
    return clip_features


def process_density_data(sd_data, segment_start, segment_end):
    density_with_time = list(zip(sd_data['time'], sd_data['y']))
    relevant_frames = []
    for data in density_with_time:
        if segment_start <= data[0] <= segment_end:
            relevant_frames.append(data)
    
    if len(relevant_frames) == 0:
        return 0
    
    sum_density = sum(data[1] for data in relevant_frames)
    return sum_density / len(relevant_frames)


def process_shots_data(sbd_data, segment_start, segment_end):
    shot_count = 0
    for shot in sbd_data["shots"]:
        shot_start, shot_end = shot["start"], shot["end"]
        ovp = max(0, min(segment_end, shot_end) - max(segment_start, shot_start)) / (segment_end - segment_start)
        if ovp > 0:
            shot_count += 1
    return shot_count


## Textual Processing
def process_sentiment_data(sentiment_data, segment_start, segment_end):
    sentiments_sum = {'positive': 0, 'negative': 0, 'neutral': 0}
    count = 0
    for sentence in sentiment_data:
        ovp = max(0, min(segment_end, sentence['end']) - max(segment_start, sentence['start'])) / (sentence['end'] - sentence['start'])
        if ovp >= 0.8:
            count += 1
            for sentiment, prob in sentence['sentiment_probs']:
                sentiments_sum[sentiment] += prob
    
    if count == 0:
        return [0, 0, 0]
    
    return [sentiments_sum[sentiment] / count for sentiment in ['positive', 'negative', 'neutral']]


def process_pos_data(pos_data, segment_start, segment_end):
    for pos_seg in pos_data["speakerturn_wise"]:
        ovp = max(0, min(segment_end, pos_seg['end']) - max(segment_start, pos_seg['start'])) / (pos_seg['end'] - pos_seg['start'])
        if ovp > 0.7:
            pos_vector = pos_seg['vector']
            num_tags = sum(pos_vector)
            if num_tags == 0:
                continue
            return [val / num_tags for val in pos_vector]
    
    return [0] * len(pos_data["speakerturn_wise"][0]['vector'])


def process_ner_data(ner_data, segment_start, segment_end):
    for ner_seg in ner_data["speakerturn_wise"]:
        ovp = max(0, min(segment_end, ner_seg['end']) - max(segment_start, ner_seg['start'])) / (ner_seg['end'] - ner_seg['start'])
        if ovp > 0.7:
            ner_vector = ner_seg['vector']
            # remove events, since no events are in the dataset
            # ner_vector = np.delete(ner_vector, 4)
            ner_vector[ner_vector > 0] = 1
            # if sum(ner_vector) == 0:
            #     continue
            # ner_vector = [val / sum(ner_vector) for val in ner_vector]
            return ner_vector
    
    return [0] * (len(ner_data["speakerturn_wise"][0]['vector']))


## Contextual Processing
def image_context_similarity(imgemb_data, segment_start, segment_end, sid, segments, config):
    fps = config['fps']
    context_size = config['context_size']

    segment_start_frame = round(segment_start * fps)
    segment_end_frame = round(segment_end * fps)
    imgembs_segment = imgemb_data[segment_start_frame:segment_end_frame]
    
    imgembs_before = []
    imgembs_after = []
    for i in range(1, context_size + 1):
        if sid - i >= 0:
            start = segments[sid - i]['start']
            end = segments[sid - i]['end']
            start_frame = round(start * fps)
            end_frame = round(end * fps)
            imgembs_before.append(imgemb_data[start_frame:end_frame])
        else:
            imgembs_before.append([])
        if sid + i < len(segments):
            start = segments[sid + i]['start']
            end = segments[sid + i]['end']
            start_frame = round(start * fps)
            end_frame = round(end * fps)
            imgembs_after.append(imgemb_data[start_frame:end_frame])
        else:
            imgembs_after.append([])
    
    before_img_similarities = []
    after_img_similarities = []
    for imgembs_context in imgembs_before:
        if len(imgembs_context) > 0:
            similarities = cosine_similarity(imgembs_context, imgembs_segment)
            before_img_similarities.append(np.quantile(similarities.flatten(), 0.8))
        else:
            before_img_similarities.append(0)
    
    for imgembs_context in imgembs_after:
        if len(imgembs_context) > 0:
            similarities = cosine_similarity(imgembs_context, imgembs_segment)
            after_img_similarities.append(np.quantile(similarities.flatten(), 0.8))
        else:
            after_img_similarities.append(0)

    return before_img_similarities + after_img_similarities

def get_matching_textsegment(textemb_data, segment_start, segment_end):
    for text_segment in textemb_data:
        text_start, text_end = text_segment['start_time'], text_segment['end_time']
        ovp = max(0, min(segment_end, text_end) - max(segment_start, text_start)) / (text_end - text_start)
        if ovp > 0.7:
            return text_segment["sentence_embeddings"]
    return []

def text_context_similarity(textemb_data, segment_start, segment_end, sid, segments, config):
    context_size = config['context_size']
    reference_segment = get_matching_textsegment(textemb_data, segment_start, segment_end)
    
    if len(reference_segment) == 0:
        return [0] * (2 * context_size)
    
    textembs_before = []
    textembs_after = []
    for i in range(1, context_size + 1):
        if sid - i >= 0:
            start = segments[sid - i]['start']
            end = segments[sid - i]['end']
            textembs_before.append(get_matching_textsegment(textemb_data, start, end))
        else:
            textembs_before.append([])
        if sid + i < len(segments):
            start = segments[sid + i]['start']
            end = segments[sid + i]['end']
            textembs_after.append(get_matching_textsegment(textemb_data, start, end))
        else:
            textembs_after.append([])

    before_text_similarities = []
    after_text_similarities = []
    for textembs_context in textembs_before:
        if len(textembs_context) > 0:
            similarities = cosine_similarity(reference_segment, textembs_context)
            before_text_similarities.append(np.max(similarities.flatten()))
        else:
            before_text_similarities.append(0)
    for textembs_context in textembs_after:
        if len(textembs_context) > 0:
            similarities = cosine_similarity(reference_segment, textembs_context)
            after_text_similarities.append(np.max(similarities.flatten()))
        else:
            after_text_similarities.append(0)

    return before_text_similarities + after_text_similarities


def segment_based_aggregation(speaker_start, speaker_end, clip_data, sbd_data, 
                              sd_data, imgemb_data, sentiment_data, pos_data, 
                              ner_data, textemb_data, sid, segments, config):
    vector = []
    
    clip_features = process_clip_data(clip_data, speaker_start, speaker_end, config)
    for clip_feature_vec in clip_features:
        vector.extend(clip_feature_vec)

    vector.append(speaker_end - speaker_start)
    vector.append(process_density_data(sd_data, speaker_start, speaker_end))
    vector.append(process_shots_data(sbd_data, speaker_start, speaker_end))
    vector.extend(process_sentiment_data(sentiment_data, speaker_start, speaker_end))
    vector.extend(process_pos_data(pos_data, speaker_start, speaker_end))
    vector.extend(process_ner_data(ner_data, speaker_start, speaker_end))
    vector.extend(image_context_similarity(imgemb_data, speaker_start, speaker_end, sid, segments, config))
    vector.extend(text_context_similarity(textemb_data, speaker_start, speaker_end, sid, segments, config))

    return vector


def window_based_aggregation(speaker_start, speaker_end, clip_data, sbd_data,
                            sd_data, imgemb_data, sentiment_data, pos_data,
                            ner_data, textemb_data, sid, segments, config):
    segment_features = []

    speaker_start_fps = round(speaker_start * config['fps']) / config['fps']
    speaker_end_fps = round(speaker_end * config['fps']) / config['fps']

    for window_length in config['window_lengths']:
        windows = []
        window_start = speaker_start_fps
        while window_start < speaker_end_fps:
            window_end = window_start + window_length
            if window_end > speaker_end_fps:
                window_end = speaker_end_fps
            windows.append((window_start, window_end))
            window_start += window_length 

        window_features = []
        for window in windows:
            window_start, window_end = window
            feature_dict = {}
            feature_dict['clip'] = process_clip_data(clip_data, window_start, window_end, config)
            feature_dict['shot_density'] = process_density_data(sd_data, window_start, window_end)
            feature_dict['sentiments'] = process_sentiment_data(sentiment_data, window_start, window_end)
            feature_dict['shots'] = process_shots_data(sbd_data, window_start, window_end)
            window_features.append(feature_dict)

        clip_features = [f['clip'] for f in window_features]
        shotdensities = [f['shot_density'] for f in window_features]
        sentiments = [f['sentiments'] for f in window_features]
        shot_counts = [f['shots'] for f in window_features]

        num_clip_domains = len(clip_features[0])
        num_windows = len(windows)
        for i in range(num_clip_domains):
            domain_vectors = [clip_features[j][i] for j in range(num_windows) if len(clip_features[j][i]) > 0]
            avgs = np.mean(domain_vectors, axis=0)
            segment_features.extend(avgs.tolist())
        
        segment_features.append(np.mean(shotdensities))
        segment_features.extend(np.quantile(list(zip(*sentiments)), 0.8, axis=1))
        segment_features.append(np.mean(shot_counts))

    segment_features.append(speaker_end - speaker_start)  # Length of speech
    segment_features.extend(process_pos_data(pos_data, speaker_start, speaker_end))
    segment_features.extend(process_ner_data(ner_data, speaker_start, speaker_end))
    segment_features.extend(image_context_similarity(imgemb_data, speaker_start, speaker_end, sid, segments, config))
    segment_features.extend(text_context_similarity(textemb_data, speaker_start, speaker_end, sid, segments, config))

    return segment_features

def main():
    parser = argparse.ArgumentParser(description='Aggregate features for the ground truth speaker turns')
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Feature Processing
    videos = args.videos
    for vi, video_path in enumerate(videos):
        print(f"Processing video {vi+1}/{len(videos)}: {video_path}", flush=True)

        output_dir = args.pkl_dir
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if os.path.exists(os.path.join(output_dir, video_name, "srr_nsr_features.pkl")):
            print(f"Features exist for video {video_name}, Skipping...", flush=True)
            continue

        # Audio Features
        asr_data = load_pickle(os.path.join(output_dir, video_name, "asr_whisperx.pkl"))["output_data"]
        # Visual Features
        clip_data = load_pickle(os.path.join(output_dir, video_name, "clip_qas.pkl"))
        sbd_data = load_pickle(os.path.join(output_dir, video_name, "transnet_shotdetection.pkl"))["output_data"]
        sd_data = load_pickle(os.path.join(output_dir, video_name, "shot_density.pkl"))["output_data"]
        imgemb_data = load_pickle(os.path.join(output_dir, video_name, "clip_embeddings.pkl"))
        # Textual Features
        sentiment_data = load_pickle(os.path.join(output_dir, video_name, "icmr_sentiment.pkl"))
        pos_data = load_pickle(os.path.join(output_dir, video_name, "whisperx_ner.pkl"))["output_data"]
        ner_data = load_pickle(os.path.join(output_dir, video_name, "whisperx_pos.pkl"))["output_data"]
        textemb_data = load_pickle(os.path.join(output_dir, video_name, "sentence_embeddings.pkl"))

        # Speaker Turns
        speaker_turns = asr_data["speaker_turns"]

        feature_turns = []

        for sid, segment in enumerate(speaker_turns):
            
            speaker_start, speaker_end = segment['start'], segment['end']

            if speaker_end - speaker_start < 1:
                vector_seg = [0] * 53
                vector_sw = [0] * 137
            else:
                vector_seg = segment_based_aggregation(speaker_start, speaker_end, clip_data, sbd_data, 
                                        sd_data, imgemb_data, sentiment_data, pos_data, 
                                        ner_data, textemb_data, sid, speaker_turns, config)
                
                vector_sw = window_based_aggregation(speaker_start, speaker_end, clip_data, sbd_data, 
                                        sd_data, imgemb_data, sentiment_data, pos_data, 
                                        ner_data, textemb_data, sid, speaker_turns, config)

            
            assert len(vector_seg) == 53
            assert len(vector_sw) == 137
            
            feature_turns.append({"id": sid, "seg_feature": np.array(vector_seg), "sw_feature": np.array(vector_sw),
                                 "start_time": speaker_start, "end_time": speaker_end})

        # Save the feature dictionary
        feature_file = os.path.join(output_dir, video_name, "srr_nsr_features.pkl")
        with open(feature_file, 'wb') as pkl:
            pickle.dump({"speaker_turns": feature_turns}, pkl, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()