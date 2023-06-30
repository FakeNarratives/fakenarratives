import argparse
import logging
import os
import pickle
from pympi.Elan import Eaf, to_eaf
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-i", "--input", type=str, help="input folder")
    parser.add_argument("-o", "--output", type=str, help="output folder")

    parser.add_argument("--shots", action="store_true", help="save shots into eaf")
    parser.add_argument("--scenes", action="store_true", help="save scenes into eaf")
    parser.add_argument("--speakers", action="store_true", help="save speakers into eaf")

    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    args = parser.parse_args()
    return args


def create_elan_file(video_name):
    eaf = Eaf(author="")
    eaf.remove_tier("default")
    eaf.add_linked_file(file_path=f"{video_name}.mp4", mimetype="video/mp4")

    return eaf


def create_shot_tier(eaf, input_pickle):
    logging.info(f"Reading: {input_pickle}")

    # create tier
    tier = "shots"
    eaf.add_tier(tier_id=tier)

    # loop through shots and add annotations
    shots = pickle.load(open(input_pickle, "rb"))
    i = 0
    for shot in enumerate(shots["output_data"]["shots"]):
        logging.debug(shot)

        start = int(1000 * shot["start"])
        end = int(1000 * shot["end"])

        if start == end:
            continue

        eaf.add_annotation(tier, start=start, end=end, value=str(i))
        i += 1

    return eaf


def create_scene_tier(input_pkl):
    # TODO
    pass


def create_speaker_tier(input_pkl):
    # TODO
    pass


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    videoname = os.path.basename(args.input)

    # create elan object
    eaf = create_elan_file(video_name=videoname)

    # add shot tier
    if args.shots:
        if "transnet_shotdetection.pkl" not in os.listdir(args.input):
            logging.error(f"Cannot find transnet_shotdetection.pkl for {videoname}")
        else:
            eaf = create_shot_tier(eaf, os.path.join(args.input, "transnet_shotdetection.pkl"))

    # write elan file
    print(videoname)
    output_file = os.path.join(args.output, f"{videoname}.eaf")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logging.info(f"Writing: {output_file}")
    to_eaf(file_path=output_file, eaf_obj=eaf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
