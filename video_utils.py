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
        scale = args.max_dimension / max(original_width, original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
    else:
        new_width, new_height = original_width, original_height

    if original_fps != required_fps:
        num_output_frames = int(total_frames * required_fps / original_fps)
        frame_indices = np.linspace(0, total_frames - 1, num=num_output_frames, dtype=int)
    else:
        frame_indices = None

    class CustomVideoReader:
        def __init__(self, video_path, width, height, frame_indices=None):
            if width == 48: # TransNetV2
                self.vr = VideoReader(video_path, num_threads=4, ctx=cpu(0))
            else:
                self.vr = VideoReader(video_path, num_threads=4, ctx=cpu(0), width=width, height=height)
            self.frame_indices = frame_indices
            self.total_frames = len(frame_indices) if frame_indices is not None else len(self.vr)
            self.width = width

        def __getitem__(self, idx):
            frame_idx = self.frame_indices[idx] if self.frame_indices is not None else idx
            return cv2.resize(self.vr[frame_idx].asnumpy(), (48, 27)) if self.width == 48 else self.vr[frame_idx].asnumpy()

        def __len__(self):
            return self.total_frames
        
        def get_batch(self, indices):
            return [self.__getitem__(idx) for idx in indices]

    custom_vr = CustomVideoReader(video_path, new_width, new_height, frame_indices)
    
    return custom_vr, new_width, new_height, required_fps, original_fps