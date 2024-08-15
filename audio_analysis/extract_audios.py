import os
from pathlib import Path
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio wav files from videos using ffmpeg")
    parser.add_argument(
        "-v",
        "--videos",
        type=str,
        required=True,
        nargs="+",
        help="Path to input videos",
    )
    parser.add_argument(
        "--pkl_dir", type=str, required=True, help="Path to pkl directory"
    )
    parser.add_argument("-r", "--rewrite", action="store_true", help="Rewrite existing files")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    for video in args.videos:
        video_path = Path(video)

        output_path = os.path.join(args.pkl_dir, video_path.stem, "audio.wav")
        
        command = (
        "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads 4 -ar 16000 %s -loglevel panic"
            % (
                video_path,
                output_path,
            )
        )
        output = subprocess.call(command, shell=True, stdout=None)

        if output == 0:
            print(f"Extracted audio from {video_path} to {output_path}")
        

if __name__ == "__main__":
    main()