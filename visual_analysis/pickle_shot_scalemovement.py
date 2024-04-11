import os
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
import pickle
import argparse
import time
from visual_utils import *
import decord
from decord import VideoReader, cpu
from torch import nn
from transformers import VideoMAEModel, VideoMAEImageProcessor
from transformers import PreTrainedModel, VideoMAEConfig
from pytorchvideo.transforms import ApplyTransformToKey

from torchvision.transforms.v2 import (
    Compose,
    Lambda,
    Resize,
    Normalize,
    UniformTemporalSubsample
)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict Shot Scale and Movement using a custom multi-task VideoMAE model.")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
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


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:# print(image_features.shape, text_features.shape)
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_subclip_indices(start_time, end_time, fps, total_frames):
    start_index = int(round(start_time * fps))
    end_index = int(round(end_time * fps))
    return [i for i in range(start_index, min(end_index + 1, total_frames))]


def process_video(video_path, output_path, model, transform, device="cuda"):
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


    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))
    print("Number of shots: ", len(shot_dict["output_data"]["shots"]))

    # Load video
    vr = VideoReader(video_path+".mp4", num_threads=12, ctx=cpu(0), width=480, height=270)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    for i, ref_shot in enumerate(shot_dict["output_data"]["shots"]):
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        all_indices = get_subclip_indices(sr_time, er_time, fps, total_frames)

        all_frames = vr.get_batch(all_indices)

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

        video_feat_dict["output_data"].append({"shot": {"start": sr_time, "end": er_time},
                                               "prediction": [scale_cat, movement_cat]})

    return video_feat_dict



def main():
    args = parse_args()

    set_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    model, processor = get_model(device)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

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

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video {i+1}/{len(input_paths)}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "videoshot_scalemovement.pkl")):
            
            st_time = time.time()
            video_feat_dict = process_video(input_path, out_loc, model, test_transform, device)

            with open(os.path.join(out_loc, "videoshot_scalemovement.pkl"), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

            print("Time taken: ", time.time()-st_time)

        print("%%\n")


if __name__ == "__main__":
    main()
