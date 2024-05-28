import pickle
from tqdm import tqdm
import warnings
from typing import Any
from pathlib import Path
import argparse

import numpy as np
from mmdet.apis import DetInferencer  # type: ignore
import decord


def parse_args():
    parser = argparse.ArgumentParser(description="Generates image captions for frames sampled from videos using specified mmdetection model.")
    parser.add_argument("-m", "--model_dir", type=Path, required=True, help="Path to work-dir of mmdetection model training with `config.py` and `last_checkpoint` file and its corresponding model weights (.pth)")
    parser.add_argument("-f", "--file", type=Path, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-v", "--video", type=Path, help="Path to video that should be processed.")
    parser.add_argument("-i", "--inp_dir", type=Path, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=Path, help="Base directory for output results. Defaults to `./{model_dir.name}/{video.name}` for video input. Otherwise `{out_dir}/{video_name}`")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size for inference. Defaults to 8 (fits into 44GB GPU).")
    parser.add_argument("--save_visualization", action="store_true", help="whether to save images with bounding boxes (numpy) aside the predictions or not. Defaults to False. (not working)")
    args = parser.parse_args()
    return args

def read_video_paths(file_path: Path, base_input_dir: Path, base_output_dir: Path) -> tuple[list[Path], list[Path]]:
    """ reads video paths from file
    File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"

    Args:
        file_path (Path): Path to file with video names
        base_input_dir (Path): base directory of input videos
        base_output_dir (Path): base directory of output

    Returns:
        tuple[list[Path], list[Path]]: input paths, output paths for each video
    """
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [base_input_dir / (vp + ".mp4") for vp in video_paths], [base_output_dir / vp for vp in video_paths]


def process_video(video_path: Path, model: DetInferencer,
                    batch_size: int, save_visualization: bool) -> dict[str, Any]:
    """
    Takes a video and generates bounding boxes/ object predictions for frames sampled at 1 fps.
    prediction format: https://mmdetection.readthedocs.io/en/latest/user_guides/inference.html

    Args:
        video_path (Path): path to video
        model (DetInferencer): mmdetection DetInferencer wrapper of model.
        batch_size (int): batch size for inference.
        save_visualization (bool): whether to save images with bounding boxes aside the predictions or not.
    Returns:
        video_feat_dict (dict): dictionary containing keys [model, video, predictions, (visualization)]

    """
    FPS_t = 1

    videoreader = decord.VideoReader(str(video_path), num_threads=1, ctx=decord.cpu(0))
    #  decord.bridge.set_bridge('torch') (incompatible with mmdetection)

    total_frames = len(videoreader)
    frame_step = int(videoreader.get_avg_fps() // FPS_t)

    print(f"Total frames: {total_frames}, frame step: {frame_step}")

    video_dict = {
        "model": {
            "source_dir": model.cfg.work_dir,
        },
        "data": {
            "num_classes": model.cfg.num_classes,
            "classes": model.cfg.test_dataloader.dataset.metainfo.classes,
        },
        "video": {
            "file_path": video_path,
            "sampling_rate": FPS_t,
            "frame_step": frame_step,
            "total_frames": total_frames,
            "fps": videoreader.get_avg_fps(),
        },
        "predictions": [],
    }
    if save_visualization:
        video_dict["visualization"] = []

    frame_list = list(range(0, total_frames, frame_step))
    video_dict["frames"] = frame_list

    # Do batched inference over frame_list and combine the results indexed by frame_list
    for i in tqdm(range(0, len(frame_list), batch_size)):
        batch_frame_list = frame_list[i:i+batch_size]
        batch_frames = videoreader.get_batch(batch_frame_list).asnumpy()

        # TODO check if channels are in correct order
        batch_results = model(unstack_array(batch_frames), batch_size=batch_size, no_save_vis= not save_visualization)
        video_dict["predictions"].append(batch_results["predictions"])
        if save_visualization:
            video_dict["visualization"].append(batch_results["visualization"])

    return video_dict


def unstack_array(array: np.ndarray) -> list[np.ndarray]:
    """ unstacks an numpy array into a list of subarrays.

    Args:
        arr (np.ndarray): array with N+1 dimensions

    Returns:
        list[np.ndarray]: list of arrays with N dimensions
    """
    splitted_array = np.split(array, len(array))
    return [np.squeeze(subarray, 0) for subarray in splitted_array]


def main():
    args = parse_args()

    assert args.file or args.video, "Either a file containing video paths or a video path should be passed."

    with open(args.model_dir / "last_checkpoint", "r") as f:
        weights_path = f.readline()

    inferencer = DetInferencer(model=str(args.model_dir / "config.py"), weights=weights_path, show_progress=False)

    if args.file:
        input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)
    else:
        input_paths = [args.video]
        output_paths = [Path(args.model_dir.name) / args.video.stem]

    for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        print(f"Video {i+1}/{len(input_paths)}: {input_path}")

        if not output_path.exists():
            output_path.mkdir(parents=True)

        output_fp = output_path / "bbox_predictions.pkl"
        print(str(output_fp))
        if not output_fp.exists():
            video_feat_dict = process_video(input_path, inferencer, args.batch_size, args.save_visualization)
            with open(output_fp, "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)
        else:
            warnings.warn(f"{output_fp} already exists. The video was skipped!")


if __name__ == "__main__":
    main()