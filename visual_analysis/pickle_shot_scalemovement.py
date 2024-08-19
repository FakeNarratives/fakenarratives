import os
import sys
import pickle
import argparse
import logging
from tqdm import tqdm
from visual_utils import *
from torch import nn
from torch.nn import functional as F
from transformers import VideoMAEModel, VideoMAEImageProcessor
from transformers import PreTrainedModel, VideoMAEConfig
from pytorchvideo.transforms import ApplyTransformToKey
sys.path.append(".")
from video_utils import read_video_and_get_info
from visual_utils import set_seeds

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

from torchvision.transforms.v2 import (
    Compose,
    Lambda,
    Resize,
    Normalize,
    UniformTemporalSubsample
)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Shot Scale and Movement using a custom multi-task VideoMAE model.")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=960,
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


def get_subclip_indices(start_index, end_index, total_frames):
    return [i for i in range(start_index, min(end_index + 1, total_frames))]


def process_video(video_path, shot_dict, model, transform, args, device="cuda"):
    """
    Take VideoMAE model and predicts both shot scale and movement for each shot in the video.

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        model (CustomModel): VideoMAE model
        transform (torchvision.transforms): transforms for the model
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
                    "prediction": [predicted_scale_class, predicted_movement_class]
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
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    total_frames = len(vr)

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        all_indices = get_subclip_indices(ref_shot["start_frame"], ref_shot["end_frame"], total_frames)

        all_frames = torch.tensor(np.array(vr.get_batch(all_indices)))

        ## Transform
        inputs = transform({"video": all_frames})["video"]

        inputs = inputs.unsqueeze(0).to(device)

        ## Predict
        with torch.no_grad():
            outputs = model(inputs)
            scale_logits = outputs["scale_logits"]
            movement_logits = outputs["movement_logits"]

        scale_pred = scale_logits.argmax(dim=-1).cpu().numpy().tolist()[0]
        movement_pred = movement_logits.argmax(dim=-1).cpu().numpy().tolist()[0]

        scale_cat = model.config.scale_id2label[str(scale_pred)]
        movement_cat = model.config.movement_id2label[str(movement_pred)]

        video_feat_dict["output_data"].append({"shot": {"start": ref_shot["start"], "end": ref_shot["end"]},
                                               "prediction": [scale_cat, movement_cat]})

    return video_feat_dict



def main():
    args = parse_args()

    set_seeds(42)

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_model(device)

    ## Transforms
    test_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
            [
                Lambda(lambda x: x.permute(0, 3, 1, 2)), # T, H, W, C -> T, C, H, W
                UniformTemporalSubsample(16),
                Lambda(lambda x: x / 255.0),
                Normalize(processor.image_mean, processor.image_std),
                Resize((model.config.image_size, model.config.image_size)),
                ]
            ),
        ),
    ])

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)

        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")

        if not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot detection file in {shot_detection_path}")
            continue

        with open(shot_detection_path, "rb") as pklfile:
            shot_content = pickle.load(pklfile)

        logging.info(f"\tNumber of shots: {len(shot_content['output_data']['shots'])}")

        video_feat_dict = process_video(video_path, shot_content, model, test_transform, args, device)

        with open(os.path.join(output_dir, "videoshot_scalemovement.pkl"), "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
