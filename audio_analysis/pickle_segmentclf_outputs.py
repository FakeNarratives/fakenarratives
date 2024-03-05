import os
import pickle
from moviepy.editor import VideoFileClip
import argparse
import tempfile
import torchaudio

from audio_utils import *
from speechbrain.pretrained.interfaces import foreign_class
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Runs speech emotion and gender classification on speakder diarization segments")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    parser.add_argument("-r", "--rewrite", action="store_true", help="Rewrite existing files")
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
    gen_model.eval()

    return emo_model, gen_model, gen_proc


def classify_asr_segments(script_dir, video_path, video, speaker_segments, 
                          emo_model, gen_model, gen_proc, emo_lab_map, 
                          gen_lab_map, device):
    """
    Takes ASR segments and Speaker Diarization segments, classifies them into speech emotion and gender categories.

    Args:
        script_dir (str): Full path of this script
        video_path (str): Full path to the video file
        video (VideoFileClip): VideoFileClip object from moviepy
        speaker_segments (list): List of speaker segments from ASR
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

    required_sr = 16000

    ## Merge speaker segments with the same speaker ID, called speaker turns
    speaker_segments = get_speaker_turns(speaker_segments)
    
    speaker_segment_preds = []
    for seg in speaker_segments:

        if seg["end"] <= seg["start"]:
            ## "Skipping segment with same start_sample and end_sample")
            speaker_segment_preds.append({"id": str(id), "start": seg["start"], "end": seg["end"], 
                                          "speaker": seg["speaker"], "gender_pred": None, "gender_prob": None, 
                                            "emotion_pred_top3": None, "emotion_prob_top3": None})
            continue
        
        
        # print("Processing segment:", seg["start"], seg["end"], seg["speaker"])

        audio_data = video.audio.subclip(seg["start"], seg["end"])

        # Manually generate a temporary file path
        with tempfile.NamedTemporaryFile(dir=os.path.join(script_dir, "temp/"), suffix=".wav", delete=True) as tmp:
        
            audio_data.write_audiofile(tmp.name, fps=required_sr, verbose=False, logger=None)
            waveform, _ = torchaudio.load(tmp.name)
            if waveform.shape[0] > 1:  # if stereo
                waveform = waveform.mean(dim=0)  # convert to mono

            
            ## Chop audio into further segments of 10 seconds
            audio_segments = torch.tensor(np.array(chop_audio_segments(waveform, required_sr, 10))) # Chop audio into 10s segments

            input_values = gen_proc(audio_segments, sampling_rate = required_sr, return_tensors='pt').input_values
            input_values = input_values.to(device).squeeze(0)

            ## Gender Recognition
            with torch.no_grad():
                result = gen_model(input_values).logits.softmax(dim=1)
                sum_res = result.mean(dim=0)
                assert sum_res.shape[0] == 2
                max_i = sum_res.detach().cpu().argmax().numpy()
                gen_prob = sum_res.detach().cpu().max().numpy()
                gen_pred = gen_lab_map[str(max_i)]  


            ## Emotion Recognition (Not a good model, Trained on English Audio)
            _, emo_prob, _, text_lab = emo_model.classify_batch(audio_segments) ## Returns all probabilities, top probability, index, label
            if len(text_lab) > 1:
                top_probs = emo_prob.detach().cpu().topk(min(len(text_lab), 3)).values.numpy().tolist()
                top_preds = [emo_lab_map[text_lab[i]] for i in emo_prob.detach().cpu().topk(min(len(text_lab), 3)).indices.numpy().tolist()]
            else:
                top_probs = emo_prob.item()
                top_preds = emo_lab_map[text_lab[0]]

            speaker = seg["speaker"] if "speaker" in seg else "None"

            speaker_segment_preds.append({"start": seg["start"], "end": seg["end"], 
                                            "speaker": speaker, "gender_pred": gen_pred, "gender_prob": gen_prob, 
                                            "emotion_pred_top3": top_preds, "emotion_prob_top3": top_probs})

                
    video_feat_dict["output_data"] = speaker_segment_preds

    return video_feat_dict, speaker_segments


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seeds(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    emo_lab_map = {"neu": "Neutral", "hap": "Happy", "ang": "Angry", "sad": "Sad"}
    gen_lab_map = {"0" : "Female", "1": "Male"}

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    emo_model, gen_model, gen_processor = get_models(device)

    if not os.path.exists(os.path.join(script_dir, "temp/")):
        os.makedirs(os.path.join(script_dir, "temp/"))

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video [{i+1}/{len(input_paths)}]\t{input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not args.rewrite:
            print("Already processed. Skipping...")
            continue

        video = VideoFileClip(input_path+".mp4")

        whisper_spk_segments = pickle.load(open(os.path.join(out_loc, "asr_whisper.pkl"), "rb"))["output_data"]["speaker_segments"]
        whisperx_spk_segments = pickle.load(open(os.path.join(out_loc, "asr_whisperx.pkl"), "rb"))["output_data"]["speaker_segments"]

        video_feat_dict, speaker_segments = classify_asr_segments(script_dir, input_path, video, whisper_spk_segments, 
                                                emo_model, gen_model, gen_processor, 
                                                emo_lab_map, gen_lab_map, device)
        

        print("Number of features:", len(video_feat_dict["output_data"]), ", Number of speaker turns:", len(speaker_segments))
        assert len(video_feat_dict["output_data"]) == len(speaker_segments)

        
        video_feat_dict2, speaker_segments2 = classify_asr_segments(script_dir, input_path, video, whisperx_spk_segments, 
                                                emo_model, gen_model, gen_processor, 
                                                emo_lab_map, gen_lab_map, device)
        
        print("Number of features:", len(video_feat_dict2["output_data"]), ", Number of speaker turns:", len(speaker_segments2))
        assert len(video_feat_dict2["output_data"]) == len(speaker_segments2)
        
        print("Saving to", os.path.join(out_loc, "whisperspeaker_segmentClf.pkl"))
        with open(os.path.join(out_loc, "whisperspeaker_segmentClf.pkl"), "wb") as f:
            pickle.dump(video_feat_dict, f)

        print("Saving to", os.path.join(out_loc, "whisperxspeaker_segmentClf.pkl"))
        with open(os.path.join(out_loc, "whisperxspeaker_segmentClf.pkl"), "wb") as f:
            pickle.dump(video_feat_dict2, f)

        print()


if __name__ == "__main__":
    main()
