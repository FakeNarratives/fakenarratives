import os
import sys
import pickle
import argparse
import logging
from tqdm import tqdm
from visual_utils import *
from torchvision.transforms import v2
from transformers import AutoModelForImageClassification, AutoImageProcessor
sys.path.append(".")
from video_utils import read_video_and_get_info
from visual_utils import set_seeds

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

def parse_args():
    parser = argparse.ArgumentParser(description="Predict camera angle or level for frames in each shot using a cinescale trained model")
    parser.add_argument("-t", "--task", type=str, default="angle", help="Perform camera angle or level classification")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=640,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


def get_model(task, device):
    if task == "angle":
        model = AutoModelForImageClassification.from_pretrained("gullalc/convnextv2-base-22k-384-cinescale-angle")
    else:
        model = AutoModelForImageClassification.from_pretrained("gullalc/convnextv2-base-22k-224-cinescale-level")

    model.eval()
    model.to(device)
    
    return model



def get_subclip_indices(start_index, end_index, fps, total_frames):
    ## Select frames at 2 fps
    return [i for i in range(start_index, min(end_index + 1, total_frames), int(fps/2))]
    # return [i for i in range(start_index, min(end_index + 1, total_frames))]


def process_video(video_path, shot_dict, model, transform, args, device="cuda"):
    """
    Take shots from the video and predicts camera angle or level for each frame in the shot using a cinescale trained model.

    Args:
        video_path (str): path to video
        output_path (str): path to load shot output
        transform (torchvision.transforms): transforms for the model
        model (AutoModelForImageClassification): cinescale model
        device (str, optional): device to run inference on. Defaults to "cuda".

    Returns:
        video_feat_dict (dict): dictionary containing shot predictions for each frame in the shot.
        As:
        {
            "github_repo": "gullalc/convnextv2-base-22k-384-cinescale-angle;https://huggingface.co/gullalc/convnextv2-base-22k-224-cinescale-level",
            "commit_id": "1e4f500c78b7ec839b8b90afee9e3efd19070f31;a4a4902c1859a876c0ad6040f034b40722d72944",
            "parameters": "default",
            "video_file": video_path,
            "output_data": [        ## List of shots
                {
                    "shot": {
                        "start": 0.0,
                        "end": 1.0
                    },
                    "predictions": [...], # Predictions for each frame (2 fps) in the shot
                }
            ]
        }
    """

    video_feat_dict = {"github_repo": "gullalc/convnextv2-base-22k-384-cinescale-angle;https://huggingface.co/gullalc/convnextv2-base-22k-224-cinescale-level",
                        "commit_id": "1e4f500c78b7ec839b8b90afee9e3efd19070f31;a4a4902c1859a876c0ad6040f034b40722d72944",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": []
                        }

    # Load video
    vr, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, 25)
    logging.info(f"\tVideo info: {len(vr)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
    total_frames = len(vr)

    for i, ref_shot in enumerate(tqdm(shot_dict["output_data"]["shots"])):
        # Each reference shot has a start time and end time
        ## Need to get the frames for the reference shot
        ## Get frames at 2 fps
        frame_indices = get_subclip_indices(ref_shot["start_frame"], ref_shot["end_frame"], fps, total_frames)
        
        ## Change N, H, W, C -> N, C, H, W
        frames = torch.tensor(np.array(vr.get_batch(frame_indices))).permute(0,3,1,2)

        ## Predict for each frame
        inputs = transform(frames).to(device)

        ## Process 32 frames at a time in a batch
        batch = []
        for i in range(0, len(inputs), 32):
            batch.append(inputs[i:i+32])

        all_preds = []
        with torch.no_grad():
            for b in batch:
                outputs = model(b).logits
                predictions = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
                
                all_preds.extend(model.config.id2label[p] for p in predictions)

        video_feat_dict["output_data"].append({"shot": ref_shot, "predictions": all_preds})

    return video_feat_dict


def main():
    args = parse_args()

    set_seeds(42)

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.task, device)

    ## Transforms
    norm_transform = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.task == "angle":                              
        test_transform = v2.Compose([v2.Resize(384, antialias=True), 
                                        v2.CenterCrop((384,384)),
                                        v2.ToDtype(torch.float32, scale=True),
                                        norm_transform])
    else:
        test_transform = v2.Compose([v2.Resize((224,224), antialias=True),
                                        v2.ToDtype(torch.float32, scale=True),
                                        norm_transform])
        

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

        with open(os.path.join(output_dir, "videoshot_%s.pkl"%(args.task)), "wb") as output_file:
            pickle.dump(video_feat_dict, output_file)

        print("%%\n")


if __name__ == "__main__":
    main()
