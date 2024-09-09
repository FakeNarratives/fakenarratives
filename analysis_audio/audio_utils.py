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

def get_speaker_turns(speaker_segments, gap=0.01):
    speaker_turns = []
    for segment in sorted(speaker_segments, key=lambda x: x["start"]):
        spk_turn_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", "").strip(),
            "speaker": segment.get("speaker", "Unknown"),
            "n_words": len(segment.get("text", "").split())
        }
        
        if not speaker_turns:
            speaker_turns.append(spk_turn_segment)
        else:
            last_turn = speaker_turns[-1]
            if last_turn["speaker"] == spk_turn_segment["speaker"]:
                last_turn["end"] = spk_turn_segment["end"]
                last_turn["text"] += " " + spk_turn_segment["text"]
                last_turn["n_words"] += spk_turn_segment["n_words"]
            else:
                if spk_turn_segment["start"] - last_turn["end"] <= gap:
                    spk_turn_segment["start"] = last_turn["end"] + gap
                speaker_turns.append(spk_turn_segment)
    
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