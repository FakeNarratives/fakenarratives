## Setup environment
conda env create -f environment.yml

## Run ASR and speaker diarization

`python text_analysis/pickle_textnlp_outputs.py -f ../video_list.txt`

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
- "output_data" has three features extracted: Sentiment, PoS Tags and Named entities
- ```
    "output_data": {"sentiment": {"sentence_wise": {}, "segment_wise": {}, "speakerseg_wise": {}},
                                "pos": {"sentence_wise": {}, "segment_wise": {}, "speakerseg_wise": {}},
                                "ner": {"sentence_wise": {}, "segment_wise": {}, "speakerseg_wise": {}}}
    ```
