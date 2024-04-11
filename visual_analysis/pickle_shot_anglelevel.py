import os
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'
import pickle
import argparse
import time
from visual_utils import *
import decord
from decord import VideoReader, cpu
from torchvision.transforms import v2
from transformers import AutoModelForImageClassification, AutoImageProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Predict camera angle or level for frames in each shot using a cinescale trained model")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-t", "--task", type=str, default="angle", help="Perform camera angle or level classification")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
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


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:# print(image_features.shape, text_features.shape)
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def get_subclip_indices(start_time, end_time, fps, total_frames):
    start_index = int(round(start_time * fps))
    end_index = int(round(end_time * fps))
    ## Select frames at 2 fps
    return [i for i in range(start_index, min(end_index + 1, total_frames), int(fps/2))]
    # return [i for i in range(start_index, min(end_index + 1, total_frames))]


def process_video(video_path, output_path, model, transform, device="cuda"):
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


    shot_dict = pickle.load(open(os.path.join(output_path, "transnet_shotdetection.pkl"), "rb"))
    print("Number of shots: ", len(shot_dict["output_data"]["shots"]))

    # Load video
    vr = VideoReader(video_path+".mp4", num_threads=4, ctx=cpu(0), width=480, height=270)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    for i, ref_shot in enumerate(shot_dict["output_data"]["shots"]):
        # Each reference shot has a start time and end time
        ## Need to get the frames for the reference shot
        ## Get frames at 2 fps
        sr_time, er_time = ref_shot["start"], ref_shot["end"]
        frame_indices = get_subclip_indices(sr_time, er_time, fps, total_frames)
        
        ## Change N, H, W, C -> N, C, H, W
        frames = vr.get_batch(frame_indices).permute(0,3,1,2)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decord.bridge.set_bridge('torch')

    model = get_model(args.task, device)

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

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
        

    for i, input_path in enumerate(input_paths):
        print(f"Processing Video {i+1}/{len(input_paths)}: {input_path}")

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        if not os.path.exists(os.path.join(out_loc, "videoshot_%s.pkl"%(args.task))):
            
            st_time = time.time()

            video_feat_dict = process_video(input_path, out_loc, model, test_transform, device)

            with open(os.path.join(out_loc, "videoshot_%s.pkl"%(args.task)), "wb") as output_file:
                pickle.dump(video_feat_dict, output_file)

            print("Time taken: ", time.time()-st_time)

        print("%%\n")


if __name__ == "__main__":
    main()
