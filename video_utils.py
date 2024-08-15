from decord import VideoReader, cpu
import cv2
import numpy as np

def read_video_and_get_info(video_path, args, required_fps):
    vr = VideoReader(video_path, num_threads=4, ctx=cpu(0))
    
    original_width, original_height = vr[0].shape[1], vr[0].shape[0]
    original_fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Resize if the max dimension is exceeded
    if max(original_width, original_height) > args.max_dimension:
        scale = args.max_dimension / max(original_width, original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
    else:
        new_width, new_height = original_width, original_height

    # Calculate frame indices using linspace
    if original_fps != required_fps:
        num_output_frames = int(total_frames * required_fps / original_fps)
        frame_indices = np.linspace(0, total_frames - 1, num=num_output_frames, dtype=int)
    else:
        frame_indices = None

    class CustomVideoReader:
        def __init__(self, vr, width, height, frame_indices=None):
            self.vr = vr
            self.frame_indices = frame_indices
            self.total_frames = len(frame_indices) if frame_indices is not None else len(vr)
            self.width = width
            self.height = height

        def __getitem__(self, idx):
            frame_idx = self.frame_indices[idx] if self.frame_indices is not None else idx
            return cv2.resize(self.vr[frame_idx].asnumpy(), dsize=(self.width, self.height))

        def __len__(self):
            return self.total_frames

    custom_vr = CustomVideoReader(vr, new_width, new_height, frame_indices)
    
    return custom_vr, new_width, new_height, required_fps