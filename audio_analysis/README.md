## Setup environment
conda env create -f environment.yml
pip install -r requirements.txt

## Create a config.yml as:

huggingface:
  token: "YOUR_HUGGINGFACE_TOKEN"

whisperx:
  batch_size: 16

## Run ASR and speaker diarization

`python audio_analysis/pickle_asr_outputs.py -f ../video_list.txt`

## Download BEATs unilm model from here and save in `pretrained_utils`:

[Beats Model](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D)

## Video list file format:

`video_list.txt` is a file with video name on each line as:


```
Tagesschau/TV-20220101-2019-5100.webl.h264
Tagesschau/TV-20220105-2021-3100.webl.h264
Tagesschau/TV-20220110-2020-5400.webl.h264
...
CompactTV/compacttv_2022_01_10_kX5DF1JDnqg
CompactTV/compacttv_2022_01_14_84f5Cfl1VPo
CompactTV/compacttv_2022_01_20_HEdDHMVfJxM
...
```


## pkl format
- Whole transcript can be accessed with ["output_data"]["text"]
- Each segment and word time stamps can be accessed in ["output_data"]["segments"]
- Speaker diarization result can be accessed in ["output_data"]["speaker_segments"]
