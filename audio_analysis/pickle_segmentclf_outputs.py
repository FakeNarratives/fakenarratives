import os
import pickle
import tempfile
from moviepy.editor import VideoFileClip
import torch
import argparse
import librosa
import soundfile as sf
from tqdm import tqdm
import torchaudio

from speechbrain.pretrained.interfaces import foreign_class
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Runs speech emotion and gender classification on ASR and speakder diarization segments")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args


def read_video_paths(file_path, base_input_dir, base_output_dir):
    ## Input base dir and output base dir are appended with video name such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_models(device):
    ## Model trained with audio sampled at 16kHz
    run_opts = {"device": device}
    emo_model = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", 
                                classname="CustomEncoderWav2vec2Classifier", run_opts=run_opts)

    ## Model trained with audio sampled at 16kHz
    gen_model = Wav2Vec2ForSequenceClassification.from_pretrained('alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')
    gen_proc = Wav2Vec2FeatureExtractor.from_pretrained('alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')
    gen_model.to(device)

    return emo_model, gen_model, gen_proc



def classify_asr_segments(video_path, asr_dict, 
                          emo_model, gen_model, 
                          gen_proc, emo_lab_map, 
                          gen_lab_map, device):
    """
    Takes ASR segments and Speaker Diarization segments, classifies them into speech emotion and gender categories.

    Args:
        video_path (str): Full path to the video file
        asr_dict (dict): ASR dictionary with transcript and speaker segments
        emo_model (Speechbrain model): Emotion recognition model from speechbrain
        gen_model (wav2vec2 model): Gender recognition model from transformers
        gen_proc (wav2vec2 processor): Gender recognition processor from transformers
        emo_lab_map: Label map for emotion recognition
        gen_lab_map: Label map for gender recognition

    Returns:
        dict: Dictionary containing:
            github_repo (str): GitHub repositories of models used separated by ;
            commit_id (str): Commit ID of the repositories (same order as github_repo) used separated by ;
            parameters (str): Parameters used for the Whisper model
            video_file (str): Full path to the video file
            output_data (dict): Segment wise classification results
    """

    video_feat_dict = {"github_repo": "https://github.com/speechbrain/speechbrain;https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
                        "commit_id": "97dbeb35bb749826583ae4c113a561b92d76d552;0a5d4dc65986030a703faf1942d51a9734824882",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": {}
                      }
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    video = VideoFileClip(video_path+".mp4")

    if not os.path.exists(os.path.join(script_dir, "temp/")):
        os.makedirs(os.path.join(script_dir, "temp/"))

    with tempfile.NamedTemporaryFile(dir=os.path.join(script_dir, "temp/"), suffix=".mp3", delete=True) as tmp:
        video.audio.write_audiofile(tmp.name)
        waveform, sample_rate = torchaudio.load(tmp.name)
        if waveform.shape[0] > 1:  # if stereo
            waveform = waveform.mean(dim=0)  # convert to mono
        waveform = waveform.to(device)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        audio_data = resampler(waveform).cpu()
        sr = 16000

        # audio_data, orig_sr = sf.read(tmp.name, dtype='float32')  # load audio and sample at 16kHz for models
        # audio_data, sr = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=16000)  # resample audio

    ## For ASR segments 
    asr_segment_preds = []

    for seg in tqdm(asr_dict["segments"]):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        slice = audio_data[start_sample:end_sample]

        input_values = gen_proc(slice, sampling_rate = 16000, padding=True, return_tensors='pt').input_values
        input_values = input_values.to(device)

        temp_filename = os.path.join(script_dir, "temp", "temp.mp3")

        with torch.no_grad():
            result = gen_model(input_values).logits
            max_i = result.detach().cpu().argmax().numpy()
            gen_pred = gen_lab_map[str(max_i)]                
        
            sf.write(temp_filename, slice, sr)

            # Perform classification
            out_prob, score, _, text_lab = emo_model.classify_file(temp_filename)

            asr_segment_preds.append({"id": seg["id"], "seek": seg["seek"], 
                                        "start": seg["start"], "end": seg["end"],
                                        "gender": gen_pred, "emotion": emo_lab_map[text_lab[0]]})
            
            # Delete the temporary file
            os.remove(temp_filename)


    ## For speaker segments 
    speaker_segment_preds = []

    for seg in tqdm(asr_dict["speaker_segments"]):
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        slice = audio_data[start_sample:end_sample]

        input_values = gen_proc(slice, sampling_rate = 16000, padding=True, return_tensors='pt').input_values
        input_values = input_values.to(device)

        with torch.no_grad():
            result = gen_model(input_values).logits
            max_i = result.detach().cpu().argmax().numpy()
            gen_pred = gen_lab_map[str(max_i)]                
        
            sf.write(temp_filename, slice, sr)

            # Perform classification
            out_prob, score, _, text_lab = emo_model.classify_file(temp_filename)

            speaker_segment_preds.append({"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"],
                                            "gender": gen_pred, "emotion": emo_lab_map[text_lab[0]]})
            
            # Delete the temporary file
            os.remove(temp_filename)
                
    video_feat_dict["output_data"]["asr_segments"] = asr_segment_preds
    video_feat_dict["output_data"]["speaker_segments"] = speaker_segment_preds

    return video_feat_dict


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emo_lab_map = {"neu": "neutral", "ang": "anger", "hap": "happy", "sad": "sad"}
    gen_lab_map = {"0" : "Female", "1": "Male"}

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    emo_model, gen_model, gen_processor = get_models(device)

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path) 

        out_loc = output_paths[i]

        if "speaker_segments" in pickle.load(open(os.path.join(out_loc, "asr.pkl"), "rb"))["output_data"]:
            if not os.path.exists(out_loc):
                os.makedirs(out_loc)

            asr_dict = pickle.load(open(os.path.join(out_loc, "asr.pkl"), "rb"))

            video_feat_dict = classify_asr_segments(input_path, asr_dict["output_data"], 
                                                    emo_model, gen_model, gen_processor, 
                                                    emo_lab_map, gen_lab_map, device)

            with open(os.path.join(out_loc, "audio_segment_classification.pkl"), "wb") as f:
                pickle.dump(video_feat_dict, f)


if __name__ == "__main__":
    main()
