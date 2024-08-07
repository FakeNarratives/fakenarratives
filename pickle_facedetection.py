import argparse
import logging
from pathlib import Path
import pickle
from tqdm import tqdm
import sys
import uuid
import cv2
import time
import torch
import insightface
from insightface.app import FaceAnalysis
from decord import VideoReader, cpu
import decord

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



def read_video_and_get_info(video_path, args):
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    ## Make longer side of the frame equal to max_dimension  if it is greater than max_dimension
    if original_width > args.max_dimension:
        new_width = args.max_dimension
        new_height = int(original_height * new_width / original_width)
    else:
        new_width = original_width
        new_height = original_height

    vr = VideoReader(video_path, num_threads=12, ctx=cpu(0), width=new_width, height=new_height)
    fps = vr.get_avg_fps()
    
    return vr, new_width, new_height, fps


def get_faces(model, vd) -> list:
    """predict all faces in a video"""
    faces = []
    for fid, frame in enumerate(tqdm(vd)):
        time = vd.get_frame_timestamp(fid)[0]
        image = frame.asnumpy()
        logging.debug(time)
        # print(time)

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
        logging.info(f"Processing video [{vi+1}/{len(videos)}]: {video_path}")
        video_path = Path(video_path)
        videoname = video_path.stem

        start = time.time()

        # get frames from video
        vd, frame_width, frame_height, fps = read_video_and_get_info(str(video_path), args)
        logging.info(f"Video info: {len(vd)} frames, {fps} FPS, {frame_width} x {frame_height}")
        args.fps = fps

        faces = get_faces(model, vd)

        logging.info(f"Total faces: {len(faces)}")
        logging.info(f"Processing time: {time.time() - start:.2f} seconds")

        # return results
        output_file = Path(args.pkl_dir, videoname, "face_detection_insightface.pkl")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "wb") as f:
            output_dict = faceanalysis_pkl(faces, args)
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {output_file}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
