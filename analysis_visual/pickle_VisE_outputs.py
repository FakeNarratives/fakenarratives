import argparse
import logging
import numpy as np
import os
import pickle
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from video_decoder import VideoBatcher, VideoDecoder
from VisE.dataset import VideoDataset
from VisE.model import ResNet50
from VisE.ontology_reader import OntologyReader


def vise_pkl(
    leaf_node_vectors: np.ndarray,
    # subgraph_vectors: np.ndarray,
    OntReader: OntologyReader,
    times: list,
    config: dict,
    args,
) -> dict:
    """Converts outputs from VisE to a pkl

    The leaf node vector indicate the similarities of 148 event types to the visual content, while the subgraph vector
    indicates the similarities of all ontology nodes including the intermediate parent nodes of the leafs

    Args:
        outputs (dict): dictionary in TIB-AV-A data.py format

    Returns:
        dict: dictionary ready to write in a .pkl
            leaf_node_vector (np.ndarray): similarities of 148 event types for all t entries in time (shape t, 148)
            leaf_node_labels (list): event type names according to Wikidata (length 148)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
            config (dict): model config
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "leaf_node_vector": leaf_node_vectors,
        "leaf_node_labels": OntReader.leaf_node_labels,
        "time": times,
        "delta_time": 1 / args.fps,
        "config": config,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # required

    # Required arguments
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--cfg", type=str, required=True, help="path to yml cfg")
    parser.add_argument("--output", type=str, required=True, help="path to the output folder")

    # optional
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument("--batch_size", type=int, required=False, default=16, help="batch size")
    parser.add_argument("--fps", type=int, required=False, default=2, help="fps to process video")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=512, help="max dimension of the video frames"
    )
    parser.add_argument(
        "--s2l_strategy",
        type=str,
        required=False,
        default="leafprob*cossim",
        choices=["leafprob", "cossim", "leafprob*cossim"],
        help="strategy to convert the subgraph vectors to leaf node vectors",
    )

    args = parser.parse_args()
    return args


def get_predictions(dataloader, OntReader, model, device, s2l_strategy):
    # subgraph_vectors = []
    leaf_node_vectors = []

    for batch in dataloader:  # loop over batches
        batch_result = model(batch["image"].to(device))
        for sample in range(batch_result["predictions"].shape[0]):  # loop over samples in batches
            # get prediction from model
            prediction = batch_result["predictions"][sample, :].detach().cpu().numpy()
            if model.model_type == "classification":
                leaf_node_vector = prediction
            elif model.model_type == "ontology":
                # convert predicted subgraph vector to leaf node vector
                leaf_node_vector = OntReader.subgraph_to_leaf_vector(
                    pred_subgraph_vector=prediction, strategy=s2l_strategy, redundancy_removal=model.redundancy_removal
                )

                if leaf_node_vector is None:  # function returned error
                    logging.error(
                        "Conversion from subgraph vector to leaf node vector failed! Correct config parameters?"
                    )
                    return {}
            else:
                logging.error("Unknown model type in cfg! Please use [classification, ontology]!")
                return {}

            # store predictions
            leaf_node_vectors.append(leaf_node_vector)
            # subgraph_vectors.append(OntReader.leaf_to_subgraph_vector(leaf_node_vector))

    return np.stack(leaf_node_vectors, axis=0)  # ,  np.stack(subgraph_vectors, axis=0)


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # load cfg
    if os.path.exists(args.cfg):
        with open(args.cfg) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            logging.debug(cfg)
    else:
        logging.error(f"Cannot find cfg file: {args.cfg}")
        return 0

    # load ontology
    OntReader = OntologyReader(
        graph_file=os.path.join(os.path.dirname(args.cfg), cfg["graph"]),
        weighting_scheme=cfg["weighting_scheme"],
        leaf_node_weight=cfg["leaf_node_weight"],
    )

    # init torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        batch_size = torch.cuda.device_count() * args.batch_size
    else:
        batch_size = args.batch_size

    # build model and load checkpoint
    if cfg["model_type"] == "ontology":
        weights = OntReader.get_node_weights(cfg["redundancy_removal"])
        num_classes = len(weights)
    else:  # cfg["model_type"] == "classification"
        num_classes = OntReader.num_leafs

    if torch.cuda.device_count() == 0:
        logging.info(f"Test on CPU with batch_size {batch_size}")
    else:
        logging.info(f"Test on {torch.cuda.device_count()} GPU(s) with batch_size {batch_size}")

    model = ResNet50(
        num_classes=num_classes, model_type=cfg["model_type"], redundancy_removal=cfg["redundancy_removal"]
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Found {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.eval()
    model.load(device=device, path=os.path.join(os.path.dirname(args.cfg), cfg["model_checkpoint"]))

    videos = args.videos

    # extract features for all videos
    for video_path in videos:
        logging.info(f"Processing video: {video_path}")
        vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=args.fps)
        vb = VideoBatcher(video_decoder=vd, batch_size=args.batch_size)

        times = []
        indices = []
        frames = []

        for batch in vb:
            times.extend(batch["time"])
            indices.extend(batch["index"])
            frames.extend(batch["frame"])

        frames = np.stack(frames, axis=0)
        logging.debug(frames.shape)
        logging.info(f"{video_path}: Get predictions for {frames.shape[0]} frames")

        # Init testing dataset
        dataset = VideoDataset(images=frames)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8)

        # predict event classes for images
        leaf_node_vectors = get_predictions(
            dataloader=dataloader,
            OntReader=OntReader,
            model=model,
            device=device,
            s2l_strategy=args.s2l_strategy,
        )
        logging.debug(leaf_node_vectors.shape)
        # logging.debug(subgraph_vectors.shape)

        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output, vidname)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, "eventclassification_vise.pkl"), "wb") as f:
            output_dict = vise_pkl(
                leaf_node_vectors=leaf_node_vectors,
                # subgraph_vectors=subgraph_vectors,
                OntReader=OntReader,
                times=times,
                config=cfg,
                args=args,
            )
            pickle.dump(output_dict, f)

        logging.info(f"Output written to: {os.path.join(output_dir, 'eventclassification_vise.pkl')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
