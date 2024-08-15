import argparse
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
import sys
import uuid
import torch
import insightface
from insightface.app import FaceAnalysis
from video_decoder import VideoDecoder


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


def get_faces(model, vd) -> list:
    """predict all faces in a video"""
    faces = []
    for sample in tqdm(vd, desc="Processing frames"):
        time = sample["time"]
        image = sample["frame"]
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
                    "frame": sample["index"],
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
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        video_path = Path(video_path)
        videoname = video_path.stem

        # get frames from video
        vd = VideoDecoder(video_path, max_dimension=args.max_dimension, fps=args.fps)
        logging.info(f"\tVideo info: {vd._frames} frames, Original FPS: {vd._real_fps}, New FPS: {vd._fps}, Size: {vd._new_size[0]} x {vd._new_size[1]}")

        faces = get_faces(model, vd)

        logging.info(f"\tDetected {len(faces)} faces in video")

        # return results
        output_file = Path(args.pkl_dir, videoname, "face_detection_insightface.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            output_dict = faceanalysis_pkl(faces, args)
            pickle.dump(output_dict, f)

        logging.info(f"\tFace detectionutput written to: {output_file}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
