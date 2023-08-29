import os
import pickle
import tempfile
from scenedetect import detect, ContentDetector
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Runs OpenAI's whisper model on news videos for automatic speech recognition and speaker diarization")
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    args = parser.parse_args()
    return args

def read_video_paths(file_path, base_input_dir, base_output_dir):
    ## Input base dir and output base dir are appended with video name such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp) for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]


def process_video(video_path):
    """
    Perfrom scene detection using pySceneDetect and save scene ids and timestamps in a pickle file

    Args:
        video_path (str): Full path to the video file

    Returns:
        video_feat_dict (dict): Dictionary containing scene ids and timestamps
    """
    video_feat_dict = {"github_repo": "https://github.com/Breakthrough/PySceneDetect",
                        "commit_id": "2af7223683c5c92fa45f9bb3656260fa823dc78c",
                        "parameters": "default",
                        "video_file": video_path,
                        "output_data": []
                      }

    scene_list = detect(video_path+".mp4", ContentDetector())

    for i, scene in enumerate(scene_list):
        temp = {}
        temp["scene_id"] = i+1
        temp["start_time"] = scene[0].get_timecode()
        temp["end_time"] = scene[1].get_timecode()
        temp["start_frame"] = scene[0].get_frames()
        temp["end_frame"] = scene[1].get_frames()
        video_feat_dict["output_data"].append(temp)

    return video_feat_dict


def main():
    args = parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, input_path in enumerate(input_paths):
        print("Video: ", i+1, input_path) 

        out_loc = output_paths[i]

        if not os.path.exists(out_loc):
            os.makedirs(out_loc)

        video_feat_dict = process_video(input_path)

        with open(os.path.join(out_loc, "scenes_scenedetect.pkl"), "wb") as f:
            pickle.dump(video_feat_dict, f)


if __name__ == "__main__":
    main()