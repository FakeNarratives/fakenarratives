import argparse
import logging
import os
import pickle
import sys

from sixdrepnet import SixDRepNet
from video_decoder import VideoBatcher, VideoDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--output", type=str, required=True, help="path to the output folder")

    # optional
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--cpu", action="store_true", help="process on cpu")
    parser.add_argument("--fps", type=int, required=False, default=2, help="fps to process video")
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=512, help="max dimension of the video frames"
    )
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # Create model
    device = -1 if args.cpu else 0
    model = SixDRepNet(gpu_id=device, dict_path="6DRepNet/model/6DRepNet_300W_LP_AFLW2000.pth")

    # loop trough input videos
    videos = args.videos
    for video_path in videos:
        logging.info(f"Processing video: {video_path}")

        # setup output dir
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output, vidname)

        if not os.path.isfile(os.path.join(output_dir, "face_analysis.pkl")):
            logging.error(f"{vidname}: face_analysis.pkl is missing")
            continue

        # read faceanalyis.pkl
        with open(os.path.join(output_dir, "face_analysis.pkl"), "rb") as pklfile:
            faces_dict = pickle.load(pklfile)

        faces = faces_dict["output_data"][0]
        for face in faces["faces"]:
            print(face.keys())
            print(face["time"])
        return

        # get frames from video
        vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=args.fps)
        vb = VideoBatcher(video_decoder=vd, batch_size=args.batch_size)

        # iterate over image batches
        times = []
        indices = []
        headposes = []

        for i, batch in enumerate(vb):
            # TODO convert iio.v3 frame into cv-like frame
            # img = cv2.imread('/path/to/image.jpg')

            pitch, yaw, roll = model.predict(batch["frame"])
            print(pitch, yaw, roll)
            headposes.append((pitch, yaw, roll))

            # store information
            times.extend(batch["time"])
            indices.extend(batch["index"])

            # model.draw_axis(img, yaw, pitch, roll)
            # cv2.imshow("test_window", img)
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
