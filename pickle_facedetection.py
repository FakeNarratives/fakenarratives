import argparse
import logging
from pathlib import Path
import pickle
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

    # optional video paramters (currently unused)
    parser.add_argument(
        "--max_dimension", default=640, type=int, help="maximum image resolution"
    )
    parser.add_argument("--fps", default=25, type=int, help="fps to read video")
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
    for frame in vd:
        time = float(frame["time"])
        image = frame["frame"]
        logging.debug(time)
        print(time)

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
                    "kps": {
                        "x": [x.item() / image_width for x in face["kps"][:, 0]],
                        "y": [y.item() / image_height for y in face["kps"][:, 1]],
                    },
                    "det_score": float(face["det_score"]),
                    "time": time,
                }
            )
        if time > 10:
            break

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
    for video_path in args.videos:
        video_path = Path(video_path)
        videoname = video_path.stem
        logging.info(f"Processing video: {video_path}")

        # get frames from video
        vd = VideoDecoder(path=video_path)
        faces = get_faces(model, vd)

        # return results
        output_file = Path(args.pkl_dir, videoname, "face_analysis.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            output_dict = faceanalysis_pkl(faces, args)
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
