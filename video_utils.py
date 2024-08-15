import torch
from decord import VideoReader, cpu
import cv2
import numpy as np

def read_video_and_get_info(video_path, args, required_fps):
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Resize if the max dimension is exceeded
    if max(original_width, original_height) > args.max_dimension:
        scale = min(args.max_dimension / max(original_width, original_height), 1)
        new_width, new_height = round(original_width * scale), round(original_height * scale)
    else:
        new_width, new_height = original_width, original_height

    # Calculate frame indices
    if original_fps != required_fps:
        frame_indices = np.arange(0, total_frames, original_fps / required_fps).astype(int)
    else:
        frame_indices = None

    class CustomVideoReader:
        def __init__(self, video_path, width, height, frame_indices=None):
            self.vr = VideoReader(video_path, num_threads=4, ctx=cpu(0), width=width, height=height)
            self.frame_indices = frame_indices
            self.total_frames = len(self.frame_indices) if frame_indices is not None else len(self.vr)

        def __getitem__(self, idx):
            if self.frame_indices is not None:
                return self.vr[self.frame_indices[idx]]
            return self.vr[idx]

        def __len__(self):
            return self.total_frames

    custom_vr = CustomVideoReader(video_path, new_width, new_height, frame_indices)
    
    return custom_vr, new_width, new_height, required_fps