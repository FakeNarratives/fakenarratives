# FakeNarratives

This repository contains code to extract auditory, textual, and visual features from videos used in the BMBF-funded project [*FakeNarratives*](https://fakenarratives.github.io)

**Table of Content**

- [FakeNarratives](#fakenarratives)
  - [Repository Layout](#repository-layout)
  - [Audio Feature Extraction](#audio-feature-extraction)
  - [Text Feature Extraction](#text-feature-extraction)
  - [Visual Feature Extraction](#visual-feature-extraction)
  - [Face Analysis](#face-analysis)
  - [Multimodal Feature Extraction](#multimodal-feature-extraction)
  - [Graph Creation](#graph-creation)

## Repository Layout

Feature extraction lives in five `analysis_*` folders, one per modality. Each `pickle_*.py` script takes one or more videos and a shot-detection pickle (from `analysis_visual/pickle_shotdetection.py`) as input, and writes one `.pkl` file per video into `results_pkl/<channel>/<video_name>/`. Most scripts share a common CLI: `-v/--videos`, `-o/--pkl_dir`, `-w/--num_workers`. Corresponding shell wrappers that batch these over a channel/corpus live in `scripts/`.

Exact output schemas (JSON structure of every `.pkl` file) are documented in [`Features.md`](Features.md) — this README only summarizes what each script does.

## Audio Feature Extraction

Code in [`analysis_audio/`](analysis_audio).

- `pickle_asr_outputs.py`: WhisperX-based ASR, diarization, and word/speaker alignment → `asr_whisperx.pkl`
- `pickle_audio_similarity.py`: audio similarity between neighboring shots (wav2vec2 / BEATs / Whisper encoders)
- `pickle_audioclf_outputs.py`: audio event classification (speech/music/silence/etc.) on shot segments
- `pickle_speakerattr_outputs.py`: speech emotion and gender classification per speaker-diarization segment
- `pickle_speakerturns_meta.py`: consolidates speaker turns with ASD, speaker roles, and news-situation labels into `speaker_turns_meta.pkl`

## Text Feature Extraction

Code in [`analysis_text/`](analysis_text).

- `pickle_texttagger.py`: POS tagging and named-entity linking on the ASR transcript
- `pickle_textsentiment.py`: sentence/turn-level sentiment classification on the ASR transcript
- `pickle_llm_evaluative.py`: LLM classification of speaker turns as evaluative/non-evaluative
- `pickle_text_features.py`: BLIP2 image captioning on sampled video frames
- `analyze_topics.py` / `visualize_topics.py`: topic modeling over transcripts and visualization
- `extract_corpus_entities.py`, `sparql_query.py`: entity extraction and Wikidata lookups

## Visual Feature Extraction

Code in [`analysis_visual/`](analysis_visual).

- `pickle_shotdetection.py`: shot boundary detection with TransNetV2 → `transnet_shotdetection.pkl` (input to most other visual/audio scripts)
- `pickle_shotdensity.py`: shot-density time series over the video
- `pickle_imageShot_similarity.py`: per-shot visual similarity to neighboring shots (SigLIP / ConvNeXtV2 / Places365)
- `pickle_actionShot_similarity.py`: per-shot action/motion similarity to neighboring shots (VideoMAE / X-CLIP)
- `pickle_layout_analysis.py`: frame layout element detection (logos, photos, backgrounds, etc.)
- `pickle_shot_anglelevel.py`: camera angle/level classification per frame
- `pickle_shot_scale.py`: shot-scale classification (VideoMAE), with `square`/`centercrop`/`fivecrop` inference variants via `-c`
- `pickle_shot_scalemovement.py`: joint shot-scale + camera-movement classification (multi-task VideoMAE), same `-c` variants
- `pickle_vlm_classification.py`: Qwen2-VL prompted classification (social roles, locations, events) at 1 FPS

## Face Analysis

Code in [`analysis_face/`](analysis_face).

- `pickle_facedetection.py`: face detection and embeddings (InsightFace)
- `pickle_facetracking.py`: face tracking across frames and crop generation
- `pickle_faceclustering.py`: identity clustering of tracked faces (DBSCAN)
- `pickle_faceemotion.py`: facial emotion classification (DeepFace)
- `pickle_headgaze.py`: head/gaze direction estimation (3DGazeNet)
- `pickle_headpose.py`: head pose estimation (6DRepNet; superseded by `pickle_headgaze.py`)
- `pickle_light-asd.py`: active speaker detection from face crops + audio (Light-ASD)
- `pickle_faceanalysis.py`: consolidates all of the above into one file per video → `face_analysis.pkl`

## Multimodal Feature Extraction

Code in [`analysis_multimodal/`](analysis_multimodal).

- `pickle_blip_features.py` / `pickle_imcap_outputs.py`: BLIP2 image captioning on sampled frames
- `srr_nsr/pickle_rolesituations.py`: speaker role and news-situation classification fusing audio/text/visual features
- `srr_nsr/textual_ftExtract.py`, `visual_ftExtract.py`: feature extraction feeding the SRR/NSR (speaker-role/news-situation) classifiers
- `srr_nsr_news_videos/`: earlier standalone SRR/NSR training pipeline (own vendored TransNetV2, feature extraction, and training/eval code); kept for reference, not part of the main per-video extraction flow

## Graph Creation

To install all dependencies, please use the following commands:

~~~sh
cd graph_visualization
conda create --name fakenarratives_graph_py311 python=3.11
pip install -r requirements.txt
~~~

To create a graphlml file based on the extracted features (from .pkl files), please run:

~~~sh
cd graph_visualization
conda activate fakenarratives_graph_py311
python export_graphml.py --input /PATH/TO/RESULT_PKLS --output /PATH/TO/OUTPUT_FOLDER
~~~
