import argparse
import logging
import numpy as np
import os
import pickle
import torch, torchvision
import sys

from semantic_geo_partitioning.geo_classification.multihead_pl_module import MultiPartClassifier
from video_decoder import VideoBatcher, VideoDecoder


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def geo_pkl(
    model: MultiPartClassifier,
    probs: list,
    geocoords: list,
    times: list,
    config: dict,
    args,
) -> dict:
    """Converts outputs from semantic_geolocation_estimation to a pkl

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            y (np.ndarray): probabilities of 14371 geographical cells for all t entries in time (shape t, 14371)
            geocoords (list): geocoordinates (lat, lng) of the most likely geographical cell for all t entries in time  (length t)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
            config (dict): model config
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    # print(model.partitionings[-1])

    # TODO: add geocoordiates for all 14,741 classes in y? -> allows geographical heatmap visualization
    return {
        "y": np.asarray(probs),
        "geocoords": geocoords,
        "time": times,
        "delta_time": 1 / args.fps,
        "config": config,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--output", type=str, required=True, help="path to the output folder")

    # argument semantic_geolocation_estimation
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./semantic_geo_partitioning/data/wacv22_checkpoints_efficientnet/semp_100_125_250/hparams.yaml",
        help="Path to hparams.yaml",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        default="./semantic_geo_partitioning/data/wacv22_checkpoints_efficientnet/semp_100_125_250/base.ckpt",
        help="Path for checkpoint file (.ckpt)",
    )
    parser.add_argument("--cpuonly", action="store_true")
    parser.add_argument("--precision", choices=[16, 32, 64], type=int, default=16)

    # optional
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="batch size")
    parser.add_argument("--fps", type=int, required=False, default=2, help="fps to process video")
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=768, help="max dimension of the video frames"
    )

    args = parser.parse_args()
    return args


def predict_geolocation(batch, model, device="cpu"):
    # preprocessing
    images = batch["frame"]

    batch_size = images.shape[0]
    images = images.transpose(0, 3, 1, 2)  # bs, c, h, w
    images_t = torch.from_numpy(images / 255.0).type(torch.float32)
    images_t = images_t.to(device)

    tfms = torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    images_norm = tfms(images_t)

    # get and stack five crops of each image
    crops = torchvision.transforms.FiveCrop(300)(images_norm)
    crops = torch.stack(crops, dim=1)

    # reshape input from [bs, crops, **] to [bs x crops, **]
    crops = torch.reshape(crops, (batch_size * 5, *crops.shape[2:]))

    # forward
    yhats = model(crops)
    yhats = yhats[-1]  # only get fine prediction

    # softmax for each crop of each partitioning
    yhats = torch.nn.functional.softmax(yhats, dim=1)

    # reshape output back to access individual crops
    yhats = torch.reshape(yhats, (batch_size, 5, *list(yhats.shape[1:])))

    # calculate max over crops
    yhats = torch.max(yhats, dim=1)[0]

    # get most likely geographical class
    pred_classes = torch.argmax(yhats, dim=1)
    pred_classes = pred_classes.cpu().detach().numpy()

    # convert most likely geographical class to coordinate
    pred_lats, pred_lngs = map(
        list,
        zip(*[model.partitionings[-1].get_lat_lng(c) for c in pred_classes.tolist()]),
    )

    geocoords = zip(pred_lats, pred_lngs)
    return yhats.cpu().detach().numpy(), geocoords


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # init torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device}")
    if torch.cuda.is_available():
        batch_size = torch.cuda.device_count() * args.batch_size
    else:
        batch_size = args.batch_size

    # init geolocation estimation model
    assert os.path.exists(args.config)
    assert os.path.exists(args.checkpoint)

    model = MultiPartClassifier.load_from_checkpoint(args.checkpoint, hparams_file=str(args.config)).eval()
    model = model.to(device)
    model.hparams["batch_size"] = args.batch_size
    videos = args.videos

    with torch.no_grad():
        # extract features for all videos
        for video_path in videos:
            logging.info(f"Processing video: {video_path}")
            # get frames from video
            vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=args.fps)
            vb = VideoBatcher(video_decoder=vd, batch_size=batch_size)

            # iterate over image batches
            times = []
            indices = []
            probs = []
            geocoords = []
            for i, batch in enumerate(vb):
                logging.debug(f"Step {i}")

                # predict geolocation
                p, c = predict_geolocation(batch=batch, model=model, device=device)

                # store information
                times.extend(batch["time"])
                indices.extend(batch["index"])
                probs.extend(p)
                geocoords.extend(c)

            # write results
            vidname = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(args.output, vidname)

            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, "semantic_geoloction_estimation.pkl"), "wb") as f:
                output_dict = geo_pkl(
                    model=model,
                    probs=probs,
                    geocoords=geocoords,
                    times=times,
                    config=model.hparams,
                    args=args,
                )
                pickle.dump(output_dict, f)
                logging.info(f"Output written to: {os.path.join(output_dir, 'semantic_geoloction_estimation.pkl')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
