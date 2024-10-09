# Audio Feature Extraction

## ASR (Automatic Speech Recognition)

### Filename: `asr_whisperx.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": {
        "text": "Transcribed text",
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "This is a sample segment."
            }
        ],
        "language": "de",
        "speaker_segments": [  // Same segments as above, assinged a speaker id
            {
                "start": 0.0,
                "end": 10.0,
                "speaker": "SPEAKER_1"
            }
        ],
        "word_segments": [
            {
                "start": 0.5,
                "end": 1.0,
                "word": "sample"
            }
        ],
        "speaker_turns": [
            {
                "start": 0.0,
                "end": 15.0,
                "speaker": "SPEAKER_1",
                "text": "",
                "n_words": 25
            }
        ]
    }
}
```

- `github_repo`: URL(s) of the GitHub repository(ies)
- `commit_id`: Specific commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path of the input video file
- `output_data`: Contains ASR and diarization results
  - `text`: Full transcription of the video
  - `segments`: List of transcription segments
  - `language`: Detected language of the transcription
  - `speaker_segments`: Speaker diarization segments
  - `word_segments`: Word-level alignment data

## Shot Audio Similarity

### Filenames: 
- `wav2vec2_audio_shot_similarity.pkl`
- `beats_audio_shot_similarity.pkl`
- `whisper_audio_shot_similarity.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "shot": {
                "start": 0.0,
                "end": 1.0
            },
            "prev_1": 0.0,
            "prev_2": 0.0,
            "next_1": 0.0,
            "next_2": 0.0
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of shot-level similarity data
  - `shot`: Details of the reference shot
  - `prev_1`, `prev_2`: Cosine similarity scores with previous shots
  - `next_1`, `next_2`: Cosine similarity scores with next shots

## Audio Type Classification

### Filenames: 
- `shot_audioClf.pkl` (Shot-level)
- `whisperxspeaker_audioClf.pkl` (Speaker-Turn level)

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "start": 0.0,
            "end": 1.0,
            "top3_label": ["speech", "music", "silence"],
            "top3_label_prob": [0.7, 0.2, 0.1]
        },
        {
            "start": 1.0,
            "end": 2.0,
            "top3_label": ["vehicle", "speech", "silence"],
            "top3_label_prob": [0.6, 0.3, 0.1]
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of shot-level or speaker-turn level classification results
  - `start`, `end`: Start and end times of the shot (in seconds)
  - `top3_label`: Top 3 predicted audio labels for the shot
  - `top3_label_prob`: Probabilities corresponding to the top 3 labels

## Speaker-Turn Speaker Audio Classification

### Filename: `whisperxspeaker_segmentClf.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "start": 0.0,
            "end": 10.0,
            "speaker": "Speaker_1",
            "gender_pred": "Male",
            "gender_prob": 0.85,
            "emotion_pred_top3": ["Neutral", "Happy", "Sad"],
            "emotion_prob_top3": [0.5, 0.3, 0.2]
        },
        {
            "start": 10.0,
            "end": 20.0,
            "speaker": "Speaker_2",
            "gender_pred": "Female",
            "gender_prob": 0.9,
            "emotion_pred_top3": ["Angry", "Sad", "Neutral"],
            "emotion_prob_top3": [0.7, 0.2, 0.1]
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the models
- `video_file`: Full path to the video file
- `output_data`: Segment-wise classification results
  - `start`, `end`: Start and end times of the speaker segment (in seconds)
  - `speaker`: Speaker ID or name for the segment
  - `gender_pred`: Predicted gender label (e.g., "Male", "Female")
  - `gender_prob`: Probability of the predicted gender
  - `emotion_pred_top3`: Top 3 predicted emotions
  - `emotion_prob_top3`: Corresponding probabilities for the top 3 emotions


# Text Feature Extraction

## Text Tagger - Part-of-Speech (PoS)

### Filename: `whisperx_pos.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": {
        "speakerturn_wise": [
            {
                "start": 0.0,
                "end": 10.0,
                "speaker": "Speaker_1",
                "tags": [
                    ["word1", "NOUN", "NN", 0, 5],
                    ["word2", "VERB", "VB", 6, 11]
                ],
                "vector": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
            }
        ],
        "pos_labelmap": {
            "ADJ": 0, "ADP": 1, "ADV": 2, "AUX": 3, "CONJ": 4, "DET": 5,
            "INTJ": 6, "NOUN": 7, "NUM": 8, "PART": 9, "PRON": 10,
            "PROPN": 11, "VERB": 12, "X": 13
        }
    }
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: Contains PoS tagging results
  - `speakerturn_wise`: List of PoS tagging results for each speaker turn
    - `start`, `end`: Start and end times of the speaker turn (in seconds)
    - `speaker`: Speaker ID or name for the segment
    - `tags`: List of tuples containing (word, universal PoS tag, language-specific PoS tag, start_char, end_char)
    - `vector`: Count vector of PoS tags according to the `pos_labelmap`
  - `pos_labelmap`: Mapping of PoS tags to indices in the count vector

## Text Tagger - Named Entity Recognition (NER)

### Filename: `whisperx_ner.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": {
        "speakerturn_wise": [
            {
                "start": 0.0,
                "end": 10.0,
                "speaker": "Speaker_1",
                "tags": [
                    {
                        "text": "Berlin",
                        "type": "LOC",
                        "start_char": 10,
                        "end_char": 16,
                        "wikidata_id": "Q64"
                        ## ... other fields
                    }
                ],
                "vector": [0, 0, 1, 0, 0, 0]
            }
        ],
        "ner_labelmap": {
            "EPER": 0, "LPER": 1, "LOC": 2, "ORG": 3, "EVENT": 4, "MISC": 5
        }
    }
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: Contains NER results
  - `speakerturn_wise`: List of NER results for each speaker turn
    - `start`, `end`: Start and end times of the speaker turn (in seconds)
    - `speaker`: Speaker ID or name for the segment
    - `tags`: List of dictionaries containing information about recognized entities
      - `text`: The recognized entity text
      - `type`: The entity type (e.g., "LOC" for location)
      - `start_char`, `end_char`: Character positions of the entity in the text
      - `wikidata_id`: Wikidata ID of the entity (if available)
    - `vector`: Count vector of NER tags according to the `ner_labelmap`
  - `ner_labelmap`: Mapping of NER tags to indices in the count vector


# Visual Feature Extraction

## Shot Detection

### Filename: `transnet_shotdetection.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": {
        "shots": [
            {
                "start_frame": 0,  // First frame of the shot
                "end_frame": 24,   // Last frame of the shot
                "start": 0.0,      // Start time of the shot in seconds
                "end": 1.0         // End time of the shot in seconds
            }
        ]
    }
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: Contains shot detection results
  - `shots`: List of detected shots, each containing start and end information in both frame numbers and seconds

## Shot Density

### Filename: `shot_density.pkl`

```json
{
    "video_file": "/path/to/video.mp4",
    "parameters": {
        "fps": 25.0,      // Frames per second used for calculation
        "bandwidth": 10.0 // Bandwidth parameter for Kernel Density Estimation
    },
    "output_data": {
        "y": [0.1, 0.2, 0.3, ...],  // Normalized shot density values (0-1 range)
        "time": [0.0, 0.04, 0.08, ...],  // Corresponding time values in seconds
        "delta_time": 0.04  // Time interval between consecutive samples (in seconds)
    }
}
```

- `video_file`: Full path to the video file
- `parameters`: Parameters used for shot density calculation
- `output_data`: Contains shot density results
  - `y`: Normalized shot density values, indicating the relative concentration of shots over time
  - `time`: Corresponding time values in seconds for each density value
  - `delta_time`: Time interval between consecutive samples, useful for reconstructing the time series

## Image-based Shot Similarity

### Filenames: 
- `siglip_shot_similarity.pkl`
- `convnextv2_shot_similarity.pkl`
- `places_shot_similarity.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "shot": {
                "start": 0.0,  // Start time of the reference shot in seconds
                "end": 1.0     // End time of the reference shot in seconds
            },
            "prev_1": [0.7, 0.75, 0.8],  // Similarity scores with the previous shot [mean, median, max]
            "prev_2": [0.6, 0.65, 0.7],  // Similarity scores with two shots before [mean, median, max]
            "next_1": [0.75, 0.8, 0.85], // Similarity scores with the next shot [mean, median, max]
            "next_2": [0.65, 0.7, 0.75]  // Similarity scores with two shots after [mean, median, max]
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of shot similarity results
  - `shot`: Start and end times of the reference shot
  - `prev_1`, `prev_2`, `next_1`, `next_2`: Similarity scores with neighboring shots
    - Each score is a list of [mean, median, max] similarity values
    - Values range from 0 to 1, where 1 indicates highest similarity

## Action-based Shot Similarity

### Filenames: 
- `kinetics-vmae_action_shot_similarity.pkl`
- `ssv2-vmae_action_shot_similarity.pkl`
- `kinetics-xclip_action_shot_similarity.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "shot": {
                "start": 0.0,  // Start time of the reference shot in seconds
                "end": 1.0     // End time of the reference shot in seconds
            },
            "prev_1": 0.7,  // Similarity score with the previous shot
            "prev_2": 0.6,  // Similarity score with two shots before
            "next_1": 0.75, // Similarity score with the next shot
            "next_2": 0.65  // Similarity score with two shots after
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of action-based shot similarity results
  - `shot`: Start and end times of the reference shot
  - `prev_1`, `prev_2`, `next_1`, `next_2`: Similarity scores with neighboring shots
    - Each score is a single float value ranging from 0 to 1, where 1 indicates highest similarity
    - These scores represent the similarity in terms of action or motion between shots

## Layout Analysis

### Filename: `layout_detection.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "start": 0.0,  // Start time of the shot in seconds
            "end": 1.0,    // End time of the shot in seconds
            "layout_cnt": {
                "background": 1.0,    // Proportion of frames (out of 5) in the shot with this layout element
                "photo_moving": 0.8, 
                "photo_still": 0.2,
                "logo": 2.0,
                // Other layout elements...
            }
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of layout analysis results for each shot
  - `start`, `end`: Start and end times of the shot in seconds
  - `layout_cnt`: Dictionary of layout elements and their proportions in the shot
    - Keys are layout element names
    - Values represent the proportion of 5 frames in the shot containing that element

## Shot Angle/Level Classification

### Filename: `videoshot_angle.pkl` or `videoshot_level.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "shot": {
                "start": 0.0,  // Start time of the shot in seconds
                "end": 1.0     // End time of the shot in seconds
            },
            "predictions": ["eye_level", "eye_level", "low_angle", ...]  // Predictions for each frame (2 fps) in the shot
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of shot angle or level classification results for each shot
  - `shot`: Start and end times of the shot in seconds
  - `predictions`: List of predicted angles or levels for each frame in the shot (sampled at 2 fps)

## Shot Scale/Movement Classification

### Filename: `videoshot_scalemovement.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "shot": {
                "start": 0.0,  // Start time of the shot in seconds
                "end": 1.0     // End time of the shot in seconds
            },
            "prediction": ["MS", "static"]  // [predicted_scale_class, predicted_movement_class]
        }
    ]
}
```

- `github_repo`: URL(s) of the GitHub repository(ies) for the model(s) used
- `commit_id`: Commit ID(s) of the model(s) used
- `parameters`: Parameters used for the model
- `video_file`: Full path to the video file
- `output_data`: List of shot scale and movement classification results for each shot
  - `shot`: Start and end times of the shot in seconds
  - `prediction`: List containing predicted scale class and movement class for the shot

## Face Analysis

### Filename: `face_analysis.pkl`

```json
{
    "faces": [
        {
            "face_id": "unique_face_id",
            "time": 0.0,        // Time of the frame in seconds
            "delta_time": 0.04, // Time interval between frames
            "frame": 0,         // Frame number
            "bbox": {           // Bounding box of the face
                "x": 0.1,
                "y": 0.1,
                "w": 0.2,
                "h": 0.2
            },
            "kpss": [...],      // Facial keypoints
            "emb": [...],       // Face embedding vector
            "track_id": "unique_track_id",
            "cluster_id": 0,    // ID of the face cluster this face belongs to
            "gaze": {
                        "left_gaze_rad": [theta_x, theta_y],
                        "right_gaze_rad": [theta_x, theta_y],
                        "left_gaze_deg": [theta_x, theta_y],
                        "right_gaze_deg": [theta_x, theta_y],
            },
            "pose": {           // Head pose information
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0
            },
            "speaking": false,  // Whether the face is speaking in this frame
            "speaking_ratio": 0.0,  // Ratio of frames/track where this face is speaking
            "speaking_frames": 0,   // Number of frames/track where this face is speaking
            "speaking_scores": [...],  // Speaking scores
            "emotions": {       // Emotion probabilities
                "angry": 0.1,
                "disgust": 0.1,
                "fear": 0.1,
                "happy": 0.5,
                "sad": 0.1,
                "surprise": 0.1,
                "neutral": 0.0
            }
        }
    ],
    "args": {...}  // Arguments used for face detection
}
```

- `faces`: List of detected and analyzed faces across all frames
  - `face_id`: Unique identifier for the face
  - `time`, `delta_time`: Timing information for the frame
  - `frame`: Frame number where the face was detected
  - `bbox`: Bounding box coordinates of the face (normalized to 0-1)
  - `kpss`: Facial keypoints
  - `emb`: Face embedding vector for recognition tasks
  - `track_id`: Unique identifier for the face track across frames
  - `cluster_id`: Identifier for the face cluster this face belongs to
  - `gaze`: Eye gaze direction (pitch and yaw angles)
  - `pose`: Head pose information (pitch, yaw, and roll angles)
  - `speaking`: Boolean indicating if the face is speaking in this frame
  - `speaking_ratio`: Proportion of frames/track where this face is speaking
  - `speaking_frames`: Total number of frames/track where this face is speaking
  - `speaking_scores`: Scores for speaking detection
  - `emotions`: Probabilities for different emotions
- `args`: Arguments used for face detection process


## Shared Object Relation [TODO]

### Filename: `shared_object_relation.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        // TODO: Add structure for shared object relation data
    ]
}
```

# Multimodal Features

## Speaker Roles & News Situations & Active Speaker Turn

### Filename: `speaker_turns_meta.pkl`

```json
{
    "github_repo": "None",
    "commit_id": "None",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        {
            "start": 0.0,           // Start time of the turn in seconds
            "end": 10.0,            // End time of the turn in seconds
            "speaker": "SPEAKER_1", // Speaker label
            "active": true,         // Boolean indicating if the turn is considered active with a speaker visible on the screen
            "active_ratio": 0.85,   // Ratio of largest track face's speaking duration to turn duration
            "active_track_id": "face_track_001", // ID of the face track with largest overlap
            "role_l0": "anchor",    // High-level speaker role
            "role_l1": "anchor",   // Detailed speaker role
            "situation": "Talking-head" // News situation category
        }
    ]
}
```

- `github_repo`: URL of the GitHub repository (if applicable)
- `commit_id`: Commit ID of the code used (if applicable)
- `parameters`: Parameters used for processing
- `video_file`: Full path to the video file
- `output_data`: List of updated speaker turns with meta information
  - `start`, `end`: Start and end times of the speaker turn in seconds
  - `speaker`: Speaker label or identifier
  - `active`: Boolean indicating if the turn is considered an active speaker turn
  - `active_ratio`: Ratio of speaking duration to turn duration (set to 0.6)
    - Value between 0 and 1, where higher values indicate more active speaking
  - `active_track_id`: Identifier of the face track with the largest overlap for this turn
    - `null` if no overlapping face track was found
  - `role_l0`: High-level speaker role category
  - `role_l1`: Detailed speaker role category
  - `situation`: News situation or context category

## LVLM [TODO]

### Filename: `lvlm_analysis.pkl`

```json
{
    "github_repo": "<URL of the GitHub repo(s) separated by ;>",
    "commit_id": "<Commit ID(s) of the model(s) used, separated by ;>",
    "parameters": "default",
    "video_file": "/path/to/video.mp4",
    "output_data": [
        // TODO: Add structure for LVLM analysis data
    ]
}
```