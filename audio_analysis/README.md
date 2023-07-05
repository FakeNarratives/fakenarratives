## Setup environment
conda env create -f environment.yml

## Install pyannote-audio
pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip

## Run ASR and speaker diarization

`python audio_analysis/pickle_asr_outputs.py -f ../video_list.txt`

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
