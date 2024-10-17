## Setup environment
conda env create -f environment.yml

## Create a config.yml as:

~~~yml
huggingface:
  token: "YOUR_HUGGINGFACE_TOKEN"
~~~

## Audio Feature Pipeline

### Extract audio.wav files for all videos - needed for most audio features

- `python audio_analysis/extract_audio.py -v <path_to_video_or_videos> -o <output_folder>`

### Run ASR and speaker diarization

- `python audio_analysis/pickle_asr_outputs.py -v <path_to_video_or_videos> -o <output_folder> -b <batch_size_for_whisperX>`

- Can increase batch size to 16 to speed up, but may run out of memory on 3090 GPU

#### Output of `asr_whisperx.pkl`:

```json
{
  "github_repo": "https://github.com/m-bain/whisperX",
  "commit_id": "f2da2f858e99e4211fe4f64b5f2938b007827e17",
  "parameters": "batch_size=<batch_size>",
  "video_file": "<video_path>",
  "output_data": {
    "language": "de",
    "text": "<full_text>",
    "segments": "<aligned_segments>",
    "speaker_segments": "<speaker_segments>",
    "word_segments": "<word_segments>",
    "speaker_turns": "<speaker_turns>"
  }
}
```


### Run Audio Classification on Shot and Speaker Turn Level

#### Download BEATs unilm model from here and save in `pretrained_utils`: [Beats Model](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)

- `python audio_analysis/pickle_audioclf_outputs.py -v <path_to_video_or_videos> -o <output_folder>`

#### Output Format of `shot_audioClf.pkl` and `whisperxspeaker_audioClf.pkl`:

```json
{
  "github_repo": "https://github.com/microsoft/unilm",
  "commit_id": "13641268b59df5cf90d27b451d87ab58b6a07055",
  "parameters": "default",
  "video_file": "<video_path>",
  "output_data": [
    {
      "start": "<start_time>",
      "end": "<end_time>",
      "top3_label": ["label_1", "label_2", "label_3"],
      "top3_label_prob": [0.9, 0.8, 0.7]
    },
    ...
  ]
}
```


### Run Speaker Attribute Classification on Speaker Turn Level

- `python audio_analysis/pickle_speakerattr_outputs.py -v <path_to_video_or_videos> -o <output_folder>`

#### Output Format of `speaker_attr.pkl`:

```json
{
  "github_repo": "https://github.com/speechbrain/speechbrain;https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
  "commit_id": "481e5dfddd70d81714b1dea32e5dbfdee7c50c03;0a5d4dc65986030a703faf1942d51a9734824882",
  "parameters": "default",
  "video_file": "<video_path>",
  "output_data": [
    {
      "start": "<start_time>",
      "end": "<end_time>",
      "speaker": "SPEAKER_01",
      "gender_pred": "<gen_pred>",
      "gender_prob": "<gen_prob>",
      "emotion_pred_top3": ["emotion_1", "emotion_2", "emotion_3"],
      "emotion_prob_top3": [0.9, 0.8, 0.7]
    },
    ...
  ]
}
```


### Run Audio Similarity on Shot Level

- `python audio_analysis/pickle_audio_similarity.py -v <path_to_video_or_videos> -o <output_folder> -m wav2vec2`

- Model `choices = ["wav2vec2", "whisper", "beats"]`

#### Output Format for `{model_type}_audio_shot_similarity.pkl`:

```json
{
  "github_repo": "",
  "commit_id": "",
  "parameters": "default",
  "video_file": "<video_path>",
  "output_data": [
    {
      "shot": {
        "start": "<start_time>",
        "end": "<end_time>"
      },
      "prev_1": 0.8,    // Similarity to previous shot
      "prev_2": 0.7,   // Similarity to 2nd previous shot
      "next_1": 0.9,  // Similarity to next shot
      "next_2": 0.6   // Similarity to 2nd next shot
    },
    ...
  ]
}
```


### Multimodal label features on speaker turn level

- These are computed after running face analysis pipeline, speaker role and news situation classification

- `python audio_analysis/pickle_speakerturns_meta.py -v <path_to_video_or_videos> -o <output_folder> -t <threshold_for_active_speech>`

#### Output Format of `speaker_turns_meta.pkl`:


```json
{
  "github_repo": "None",
  "commit_id": "None",
  "parameters": "default",
  "video_file": "<video_path>",
  "output_data": [    // Updated speaker turn data with multimodal features
    {
      "start": "<start_time>",
      "end": "<end_time>",
      "speaker": "SPEAKER",
      "active": true,   // Active speech or not
      "active_ratio": 0.75,   // Ratio of active speech
      "active_track_id": "<track_id>",  // Track ID of largest overlapping active face track
      "role_l0": "anchor",  // Role level 0 ("anchor", "reporter", "other")
      "role_l1": "reporter",  // Role level 1 ("anchor", "reporter", "expert", "layperson", "politician", "other")
      "situation": "talking-head" // News situation ("talking-head", "voiceover", "interview", "commenting", "speech")
    },
    ...
  ]
}
```

