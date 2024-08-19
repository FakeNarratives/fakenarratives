import argparse
import logging
from pathlib import Path
import pickle
import os
from tqdm import tqdm
import sys
import torch
import uuid
import time
import insightface
from insightface.app import FaceAnalysis
from video_utils import read_video_and_get_info


assert insightface.__version__ >= "0.3"


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="insightface detection and facial feature extraction"
    )
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")

    # required inputs
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

    # optional face detection parameters
    parser.add_argument("--ctx", default=0, type=int, help="ctx id, <0 means using cpu")
    parser.add_argument("--det-size", default=640, type=int, help="detection size")
    parser.add_argument("--fps", type=int, default=25, help="frames per second")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


def faceanalysis_pkl(faces: list, args: dict) -> dict:
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": faces,
        "args": args_dict,
    }


def get_faces(model, vd, args) -> list:
    """predict all faces in a video"""
    faces = []
    for fid, frame in enumerate(tqdm(vd, desc="Processing frames")):
        time = fid/args.fps
        image = frame
        logging.debug(time)

        for face in model.get(image):
            x, y = round(max(0, face["bbox"][0])), round(max(0, face["bbox"][1]))
            w, h = round(face["bbox"][2] - x), round(face["bbox"][3] - y)

            image_height = image.shape[0]
            image_width = image.shape[1]

            # store normalized bbox and kps
            faces.append(
                {
                    "id": uuid.uuid4(),
                    "bbox": {
                        "x": float(x / image_width),
                        "y": float(y / image_height),
                        "w": float(w / image_width),
                        "h": float(h / image_height),
                    },
                    "kps": [
                        [
                            float(face["kps"][i, 0] / image_width),
                            float(face["kps"][i, 1] / image_height)
                        ]
                        for i in range(face["kps"].shape[0])
                    ],
                    "det_score": float(face["det_score"]),
                    "time": time,
                    "delta_time": 1 / args.fps,
                    "frame": fid,
                    "embedding": face["embedding"].tolist(),
                }
            )

    return faces


def main() -> int:
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    # load models
    model = FaceAnalysis(allowed_modules=["detection", "recognition"])
    model.prepare(ctx_id=args.ctx, det_size=(args.det_size, args.det_size))

    # loop trough input videos
    videos = args.videos
    for vi, video_path in enumerate(videos):
        start = time.time()
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        videoname = os.path.splitext(os.path.basename(video_path))[0]

        # get frames from video
        vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.fps)
        logging.info(f"\tVideo info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")

        args.frame_width = frame_width
        args.frame_height = frame_height

        faces = get_faces(model, vd, args)

        logging.info(f"\tDetected {len(faces)} faces in video")

        # return results
        output_file = Path(args.pkl_dir, videoname, "face_detection_insightface.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            output_dict = faceanalysis_pkl(faces, args)
            pickle.dump(output_dict, f)

        logging.info(f"\tFace detection output written to: {output_file}")
        logging.info(f"\tTime taken: {time.time() - start:.2f} seconds")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
