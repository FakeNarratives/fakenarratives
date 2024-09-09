import torch.nn.functional as F
import numpy as np
import torch
import random

def exact_div(x, y):
    assert x % y == 0
    return x // y

## From Whisper Code
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

# def get_speaker_turns(speaker_segments, speaker_seg_tol=3.0):
#     for i, segment in enumerate(speaker_segments):
#         start_time = round(segment["start"], 2)
#         end_time = round(segment["end"], 2)
#         if i < len(speaker_segments) - 1:
#             if speaker_segments[i + 1]["start"] - end_time < speaker_seg_tol:
#                 end_time = round(
#                     speaker_segments[i + 1]["start"] - 0.04, 2
#                 )  # Similar to shots

#         segment["start"] = start_time
#         segment["end"] = end_time
#         segment["n_words"] = len(segment["text"].split())
#         print(start_time, end_time, segment["speaker"])
        
#         if "speaker" not in segment:
#             segment["speaker"] = "Unknown"

#     speaker_turns = []
#     current_span = None
#     for segment in speaker_segments:
#         if current_span is None or segment["speaker"] != current_span["speaker"]:
#             current_span = segment.copy()
#             speaker_turns.append(current_span)
#         else:
#             current_span["end"] = segment["end"]
#             current_span["text"] += " " + segment["text"].strip()
#             current_span["n_words"] += segment["n_words"]

#     return speaker_turns, speaker_segments


def get_speaker_turns(speaker_segments, gap=0.01):
    speaker_segments = sorted(speaker_segments, key=lambda x: x["start"])

    speaker_turns = []

    for segment in speaker_segments:

        segment["n_words"] = len(segment["text"].split()) if "text" in segment else 0

        if "speaker" not in segment or segment["speaker"] is None:
            segment["speaker"] = "Unknown"

        if not speaker_turns:
            speaker_turns.append(segment.copy())
        else:
            last_turn = speaker_turns[-1]

            ## Checking if current segment belongs to same speaker
            if last_turn["speaker"] == segment["speaker"]:
                # last_turn["end"] = max(last_turn["end"], segment["end"]) # If no gap to be add
                last_turn["end"] = max(last_turn["end"], segment["end"] - gap)   # Just to ensure some gap between turns

                last_turn["text"] += " " + segment["text"].strip()
                last_turn["n_words"] += len(segment["text"].split())
            else:
            ## Treat otherwise as a new speaker turn
                if segment["start"] - last_turn["end"] < gap:       ## Comment if no gap to add
                    segment["start"] = last_turn["end"] + gap
                
                speaker_turns.append(segment.copy())
    
    return speaker_turns


def chop_audio_segments(waveform, sampling_rate, window):
    samples_per_segment = window * sampling_rate  # 10 seconds * 16000 samples/second
    num_full_segments = waveform.shape[0] // samples_per_segment
    segments = []
    ## Full segments
    for i in range(num_full_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = waveform[start:end]
        segments.append(segment.numpy())

    # Add the remaining segment if it exists
    if waveform.shape[0] % samples_per_segment != 0:
        start = num_full_segments * samples_per_segment
        end = waveform.shape[0] 
        segment = waveform[start:end]
        num_zeroes = samples_per_segment - (end - start)
        padded_segment = F.pad(segment, (0, num_zeroes))
        segments.append(padded_segment.numpy())

    return segments


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False