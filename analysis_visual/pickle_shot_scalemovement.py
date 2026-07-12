import os
import sys
import pickle
import argparse
import logging
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.nn import functional as F
from transformers import VideoMAEModel, VideoMAEImageProcessor
from transformers import PreTrainedModel, VideoMAEConfig
from pytorchvideo.transforms import ApplyTransformToKey
sys.path.append(".")
from project_utils import read_video_and_get_info, set_seeds

from torchvision.transforms.v2 import (
    Compose,
    Lambda,
    Resize,
    Normalize,
    CenterCrop,
    FiveCrop,
    UniformTemporalSubsample
)

## Maps --crop_type to the output pickle filename
OUTPUT_FILENAMES = {
    "square": "videoshot_scalemovement.pkl",
    "centercrop": "videoshot_scalemovement_centercrop.pkl",
    "fivecrop": "videoshot_scalemovement_fivecrop.pkl",
}

## Number of frames sampled per shot for the model (must match UniformTemporalSubsample below)
NUM_TEMPORAL_SAMPLES = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Shot Scale and Movement using a custom multi-task VideoMAE model.")
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", "--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="number of workers to use for data loading")
    parser.add_argument(
        "-r", "--resume",
        action="store_true",
        help="reprocess and overwrite the output pickle even if it already exists",
    )
    parser.add_argument(
        "-c", "--crop_type",
        type=str,
        required=True,
        choices=list(OUTPUT_FILENAMES.keys()),
        help="frame preprocessing strategy: 'square' (resize to square, no crop), 'centercrop' (resize + center crop), 'fivecrop' (resize + five crop with majority vote)",
    )
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


# Custom VideoMAE multi-task model definition for shot scale and movement classification
class CustomVideoMAEConfig(VideoMAEConfig):
    def __init__(self, scale_label2id=None, scale_id2label=None, movement_label2id=None, movement_id2label=None, **kwargs):
        super().__init__(**kwargs)
        self.scale_label2id = scale_label2id if scale_label2id is not None else {}
        self.scale_id2label = scale_id2label if scale_id2label is not None else {}
        self.movement_label2id = movement_label2id if movement_label2id is not None else {}
        self.movement_id2label = movement_id2label if movement_id2label is not None else {}


class CustomModel(PreTrainedModel):
    config_class = CustomVideoMAEConfig

    def __init__(self, config, scale_num_classes=None, movement_num_classes=None):
        super().__init__(config)
        scale_num_classes = scale_num_classes or len(config.scale_label2id)
        movement_num_classes = movement_num_classes or len(config.movement_label2id)
        self.vmae = VideoMAEModel(config)
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        self.scale_cf = nn.Linear(config.hidden_size, scale_num_classes)
        self.movement_cf = nn.Linear(config.hidden_size, movement_num_classes)

    def forward(self, pixel_values, scale_labels=None, movement_labels=None):

        vmae_outputs = self.vmae(pixel_values)
        sequence_output = vmae_outputs[0]

        if self.fc_norm is not None:
            sequence_output = self.fc_norm(sequence_output.mean(1))
        else:
            sequence_output = sequence_output[:, 0]

        scale_logits = self.scale_cf(sequence_output)
        movement_logits = self.movement_cf(sequence_output)

        if scale_labels is not None and movement_labels is not None:
            loss = F.cross_entropy(scale_logits, scale_labels) + F.cross_entropy(movement_logits, movement_labels)
            return {"loss": loss, "scale_logits": scale_logits, "movement_logits": movement_logits}
        return {"scale_logits": scale_logits, "movement_logits": movement_logits}


def get_model(device):
    path = "gullalc/videomae-base-finetuned-kinetics-movieshots-multitask"
    model = CustomModel.from_pretrained(path)
    processor = VideoMAEImageProcessor.from_pretrained(path)
    model.eval()
    model.to(device)
    return model, processor


def get_transform(crop_type, image_size, processor):
    """
    Build the frame preprocessing pipeline for the given crop_type.

    'square' and 'centercrop' produce a single (T, C, S, S) tensor and are normalized
    inline. 'fivecrop' produces a tuple of 5 crops, so normalization is applied per-crop
    in process_video instead (FiveCrop can't be followed by Normalize in the same Compose).
    """
    ops = [
        Lambda(lambda x: x.permute(0, 3, 1, 2)),  # T, H, W, C -> T, C, H, W
        UniformTemporalSubsample(NUM_TEMPORAL_SAMPLES),
    ]

    if crop_type == "square":
        ops.append(Resize((image_size, image_size)))
    elif crop_type == "centercrop":
        ops.append(Resize(image_size))
        ops.append(CenterCrop((image_size, image_size)))
    elif crop_type == "fivecrop":
        ops.append(Resize(image_size))
        ops.append(FiveCrop((image_size, image_size)))
    else:
        raise ValueError(f"Unknown crop_type: {crop_type}")

    if crop_type != "fivecrop":
        ops.append(Lambda(lambda x: x / 255.0))
        ops.append(Normalize(processor.image_mean, processor.image_std))

    return Compose([ApplyTransformToKey(key="video", transform=Compose(ops))])


def get_subclip_indices(start_index, end_index, total_frames, num_samples=NUM_TEMPORAL_SAMPLES):
    """
    Pick num_samples frame indices evenly spaced across [start_index, end_index], matching
    what UniformTemporalSubsample(num_samples) would later pick from the full range. Sampling
    here (before vr.get_batch) instead of after avoids decoding every frame in the shot first
    - some shots span tens of thousands of frames (bad shot-detection splits), and decoding
    all of them before subsampling to num_samples can OOM a multi-hundred-GB host.
    """
    end_index = min(end_index, total_frames - 1)
    num_shot_frames = max(1, end_index - start_index + 1)
    positions = torch.linspace(0, num_shot_frames - 1, num_samples).long()
    return [start_index + p.item() for p in positions]


def predict_shot(inputs, model, crop_type, processor, device):
    """
    Run the model on the transformed clip for a single shot and return
    (scale_cat, cumulative_distance, movement_cat).

    For 'fivecrop', inputs is a tuple of 5 unnormalized crop tensors; predictions
    are combined across crops via majority vote, and cumulative_distance uses the
    scale probabilities averaged across crops.
    """
    if crop_type == "fivecrop":
        normed_inputs = [
            Normalize(processor.image_mean, processor.image_std)(crop.float().div(255.0))
            for crop in inputs
        ]  # list of 5 tensors, each (16,3,S,S)
        batch = torch.stack(normed_inputs, dim=0).to(device)  # (5,16,3,S,S)
    else:
        batch = inputs.unsqueeze(0).to(device)  # (1,16,3,S,S)

    with torch.no_grad():
        outputs = model(batch)
        scale_logits = outputs["scale_logits"]
        movement_logits = outputs["movement_logits"]

    scale_probs = F.softmax(scale_logits, dim=-1).cpu().numpy()  # (N, num_scales)

    if crop_type == "fivecrop":
        scale_preds = scale_logits.argmax(dim=-1)
        movement_preds = movement_logits.argmax(dim=-1)
        scale_pred, _ = torch.mode(scale_preds, dim=0)
        movement_pred, _ = torch.mode(movement_preds, dim=0)
        scale_pred = scale_pred.item()
        movement_pred = movement_pred.item()
        avg_probs = scale_probs.mean(axis=0)
        cumulative_distance = round((avg_probs * np.arange(1, avg_probs.shape[0] + 1)).sum(), 4)
    else:
        scale_pred = scale_logits.argmax(dim=-1).cpu().numpy().tolist()[0]
        movement_pred = movement_logits.argmax(dim=-1).cpu().numpy().tolist()[0]
        cumulative_distance = round(sum(scale_probs[0] * np.arange(1, scale_probs.shape[1] + 1)), 4)

    scale_cat = model.config.scale_id2label[str(scale_pred)]
    movement_cat = model.config.movement_id2label[str(movement_pred)]

    return scale_cat, cumulative_distance, movement_cat


def process_video(video_path, shot_dict, model, transform, processor, crop_type, args, device="cuda"):
    """
    Take VideoMAE model and predicts both shot scale and movement for each shot in the video.

    Args:
        video_path (str): path to video
        shot_dict (dict): loaded shot detection output
        model (CustomModel): VideoMAE model
        transform (torchvision.transforms): transforms for the model
        processor (VideoMAEImageProcessor): processor holding image_mean/image_std
        crop_type (str): 'square', 'centercrop' or 'fivecrop'
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing predicted shot scale and movement for each shot in the video.
        As:
        {
            "github_repo": "https://huggingface.co/gullalc/videomae-base-finetuned-kinetics-movieshots-multitask",
            "commit_id": "23e38eaf25a909c8f5d00c6803172e50cf2058ee",
            "parameters": "default",
            "video_file": video_path,
            "output_data": [
                {
                    "shot": {
                        "start": 0.0,
                        "end": 1.0
                    },
                    "prediction": [predicted_scale_class, cummulative_distance, predicted_movement_class]
                }
            ]
        }
    """

    video_feat_dict = {"github_repo": "https://huggingface.co/gullalc/videomae-base-finetuned-kinetics-movieshots-multitask",
                        "commit_id": "23e38eaf25a909c8f5d00c6803172e50cf2058ee",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": []
                        }

    # Load video
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.num_workers, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    total_frames = len(vr)

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        all_indices = get_subclip_indices(ref_shot["start_frame"], ref_shot["end_frame"], total_frames)

        all_frames = torch.tensor(np.array(vr.get_batch(all_indices)))

        ## Transform
        inputs = transform({"video": all_frames})["video"]

        scale_cat, cumulative_distance, movement_cat = predict_shot(inputs, model, crop_type, processor, device)

        video_feat_dict["output_data"].append({"shot": {"start": ref_shot["start"], "end": ref_shot["end"]},
                                               "prediction": [scale_cat, cumulative_distance, movement_cat]})

    return video_feat_dict



def main():
    args = parse_args()

    set_seeds(42)

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_model(device)

    output_filename = OUTPUT_FILENAMES[args.crop_type]

    ## Transforms
    test_transform = get_transform(args.crop_type, model.config.image_size, processor)

    videos = args.videos
    successful = 0
    failed = 0
    failed_videos = []

    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        if os.path.exists(os.path.join(output_dir, output_filename)) and not args.resume:
            logging.info(f"\tFound existing pickle file in {output_dir}, skipping")
            successful += 1
            continue

        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")

        if not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot detection file in {shot_detection_path}")
            failed += 1
            failed_videos.append(video_path)
            continue

        try:
            with open(shot_detection_path, "rb") as pklfile:
                shot_content = pickle.load(pklfile)

            logging.info(f"\tNumber of shots: {len(shot_content['output_data']['shots'])}")

            video_feat_dict = process_video(video_path, shot_content, model, test_transform, processor, args.crop_type, args, device)

            with open(os.path.join(output_dir, output_filename), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

            successful += 1
            print("%%\n")
        except Exception as e:
            logging.error(f"\tError processing video {video_path}: {e}")
            failed += 1
            failed_videos.append(video_path)

    total = len(videos)
    logging.info(f"Processing complete. Total: {total}, Successful: {successful}, Failed: {failed}")
    print(f"\nProcessing summary:")
    print(f"Total videos: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_videos:
        print("Failed videos:")
        for v in failed_videos:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
