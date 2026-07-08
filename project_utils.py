import os
import logging
import cv2
import torch
import random
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from decord import VideoReader, cpu
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"


def read_video_and_get_info(video_path, args, workers, required_fps):
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
                self.vr = VideoReader(video_path, num_threads=workers, ctx=cpu(0))
            else:
                self.vr = VideoReader(video_path, num_threads=workers, ctx=cpu(0), width=width, height=height)
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


def setup_logging(script_name: str, log_level: int = logging.INFO) -> Path:
    """
    Set up logging for a script.
    
    Args:
    script_name (str): Name of the script (used in the log file name)
    log_level (int): Logging level (default: logging.INFO)
    
    Returns:
    Path: Path to the created log file
    """
    # Determine the project root (assuming project_utils.py is in the root directory)
    project_root = Path(__file__).parent
    
    # Create logs directory if it doesn't exist
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Set up logging
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the root logger
    logging.getLogger('').addHandler(file_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")
    
    return log_file

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = True


def load_pickle(file_path: Path) -> Any:
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data: Any, file_path: Path) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)