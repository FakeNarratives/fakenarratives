import os
import pickle
import argparse
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from text_utils import *
import umap
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from typing import Union
import sys
sys.path.append(".")
from project_utils import set_seeds

def parse_args():
    parser = argparse.ArgumentParser(description="Topic modeling for German news videos")
    parser.add_argument("-i", "--inp_dir", type=str, required=True, help="Directory containing ASR output files")
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--min_clust_size", type=int, default=10, help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="Minimum samples for HDBSCAN")
    parser.add_argument("--diversity", type=float, default=0.7, help="Diversity parameter for MMR")
    parser.add_argument("--neighbors", type=int, default=5, help="Number of neighbors for UMAP")
    parser.add_argument("--components", type=int, default=10, help="Number of components for UMAP")
    return parser.parse_args()

    """
    UMAP: Higher neighbors leads to less topics, higher components leads to more topics
    Diversity: Higher diversity leads to more diverse topics (Range 0-1)
    """


def split_text(text: str, max_length: int = 128) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def load_asr_data(directory: str, min_words: int = 5) -> Tuple[List[str], List[Dict], Dict[str, str], Dict[str, List[Dict]]]:
    full_texts = []
    all_speaker_turns = []
    video_to_text = {}
    video_to_speaker_turns = {}

    for video in os.listdir(directory):
        vidname = os.path.splitext(video)[0]
        video_folder_path = os.path.join(directory.replace("videos", "results_pkl"), vidname)

        if not os.path.isdir(video_folder_path):
            continue

        asr_file = os.path.join(video_folder_path, "asr_whisperx.pkl")
        if not os.path.exists(asr_file):
            continue

        with open(asr_file, "rb") as f:
            data = pickle.load(f)

        full_text = data['output_data']['text']
        if isinstance(full_text, str) and len(full_text.split()) >= min_words:
            full_texts.append(full_text)
            video_to_text[vidname] = full_text

        speaker_turns = data['output_data']['speaker_turns']
        filtered_turns = [
            {'id': tid, 'start': turn['start'], 'end': turn['end'], 'text': turn['text']}
            for tid, turn in enumerate(speaker_turns) 
            if isinstance(turn['text'], str) and len(turn['text'].split()) >= min_words
        ]
        if filtered_turns:
            all_speaker_turns.extend(filtered_turns)
            video_to_speaker_turns[vidname] = filtered_turns

    print(f"Loaded {len(full_texts)} full texts and {len(all_speaker_turns)} speaker turns.")
    return full_texts, all_speaker_turns, video_to_text, video_to_speaker_turns



def create_topic_model(args):
    # Load custom stopwords
    stopwords = []
    with open("analysis_text/stopwords.txt", "r") as fr:
        for line in fr:
            stopwords.append(line.strip())

    ## paraphrase-multilingual-MiniLM-L12-v2
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    umap_model = umap.UMAP(n_neighbors=args.neighbors, n_components=args.components, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=args.min_clust_size, min_samples=args.min_samples, 
                            metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words=stopwords)
    ctfidf_model = ClassTfidfTransformer()
    # representation_model = KeyBERTInspired()
    representation_model = MaximalMarginalRelevance(diversity=args.diversity)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        language="multilingual",
        top_n_words=15,
        n_gram_range=(1, 2)
    )

    return topic_model, embedding_model


def run_topic_modeling(texts: List[Union[str, Dict]], model: BERTopic, 
                       pre_compute_embeddings: bool = False, sentence_model=None, 
                       max_length: int = 128) -> Tuple[List[int], Dict]:
    # Extract text from dictionaries if necessary
    processed_texts = [text['text'] if isinstance(text, dict) else text for text in texts]
    
    # Filter out non-string elements and empty strings
    filtered_texts = [text for text in processed_texts if isinstance(text, str) and text.strip()]
    
    if not filtered_texts:
        print("Warning: No valid texts to process after filtering.")
        return [], {}

    print(f"Processing {len(filtered_texts)} texts for topic modeling.")
    
    if pre_compute_embeddings:
        chunked_embeddings = []
        for text in filtered_texts:
            chunks = split_text(text, max_length)
            chunk_embeddings = sentence_model.encode(chunks, show_progress_bar=False)
            avg_embedding = np.mean(chunk_embeddings, axis=0)
            chunked_embeddings.append(avg_embedding)
        
        # Fit transform with pre-computed average embeddings
        topics, probs = model.fit_transform(filtered_texts, np.array(chunked_embeddings))
    else:
        # Regular fit transform for speaker turns
        topics, probs = model.fit_transform(filtered_texts)
    
    topic_info = model.get_topic_info()
    return topics, topic_info.to_dict('records')


def save_results(out_dir: str, video_topics: List[int], video_topic_info: Dict,
                 turn_topics: List[int], turn_topic_info: Dict,
                 video_to_text: Dict[str, str], video_to_speaker_turns: Dict[str, List[Dict]],
                 id_to_topic_turns: Dict[int, str]):
    os.makedirs(out_dir, exist_ok=True)

    # Save video-level topics
    video_df = pd.DataFrame({
        'video': list(video_to_text.keys()),
        'topic': video_topics
    })
    video_df.to_csv(os.path.join(out_dir, 'video_topics.csv'), index=False)

    # Save speaker turn topics
    turn_data = []
    for video, turns in video_to_speaker_turns.items():
        video_turns = []
        for i, turn in enumerate(turns):
            turn_data.append({
                'video': video,
                'turn_index': turn['id'],
                'start': turn['start'],
                'end': turn['end'],
                'topic': turn_topics[len(turn_data)] if len(turn_data) < len(turn_topics) else -1
            })

            video_turns.append(
                {
                    'turn_index': turn['id'],
                    'start': turn['start'],
                    'end': turn['end'],
                    'text': turn['text'],
                    'topic': id_to_topic_turns[turn_data[-1]['topic']]["name"],
                    'words': id_to_topic_turns[turn_data[-1]['topic']]["words"]
                }
            )

        pickle.dump(video_turns, open(os.path.join(out_dir, "pkls", f"{video}_turn_topics.pkl"), "wb"))

    turn_df = pd.DataFrame(turn_data)
    turn_df.to_csv(os.path.join(out_dir, 'speaker_turn_topics.csv'), index=False)

    # Save topic info
    pd.DataFrame(video_topic_info).drop(columns=["Representative_Docs"]).to_csv(os.path.join(out_dir, 'video_topic_info.csv'), index=False)
    pd.DataFrame(turn_topic_info).to_csv(os.path.join(out_dir, 'turn_topic_info.csv'), index=False)


def main():
    args = parse_args()

    set_seeds(42)

    # Load ASR data
    full_texts, all_speaker_turns, video_to_text, video_to_speaker_turns = load_asr_data(args.inp_dir, 25)

    print(f"Number of full texts: {len(full_texts)}")
    print(f"Number of speaker turns: {len(all_speaker_turns)}")

    # Create topic model
    topic_model, sentence_model = create_topic_model(args)

    # Run topic modeling for full videos with pre-computed embeddings
    print("Processing full video texts:")
    video_topics, video_topic_info = run_topic_modeling(full_texts, topic_model, pre_compute_embeddings=True, 
                                                        sentence_model=sentence_model, max_length=128)

    # Run topic modeling for speaker turns (no need for pre-computed embeddings)
    print("Processing speaker turn texts:")
    turn_topics, turn_topic_info = run_topic_modeling(all_speaker_turns, topic_model)

    id_to_topic_turns = {topic['Topic'] : {"name": topic['Name'], "words": topic["Representation"]} for topic in turn_topic_info}

    # Save results
    save_results(args.out_dir, video_topics, video_topic_info, turn_topics, turn_topic_info, 
                 video_to_text, video_to_speaker_turns, id_to_topic_turns)

if __name__ == "__main__":
    main()