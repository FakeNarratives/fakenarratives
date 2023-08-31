import argparse
import csv
import logging
import os
import pickle
import sys
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

sys.path.append("..")
from video_decoder import VideoDecoder


class BLIP2:
    def __init__(
        self,
        model="Salesforce/instructblip-flan-t5-xl",
        processor="Salesforce/instructblip-flan-t5-xl",
        do_sample=False,
        num_beams=1,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    ):
        self._model = InstructBlipForConditionalGeneration.from_pretrained(model)
        self._processor = InstructBlipProcessor.from_pretrained(processor)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

        self._params = {
            "do_sample": do_sample,
            "num_beams": num_beams,
            "max_length": max_length,
            "min_length": min_length,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "temperature": temperature,
        }

    def get_response(self, image, query):
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self._device)
        outputs = self._model.generate(**inputs, **self._params)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def geo_pkl(
    probs: list,
    responses: list,
    times: list,
    config: dict,
    args,
) -> dict:
    """Converts outputs from instructblip to a pkl

    Returns:
        dict: dictionary ready to write in a .pkl
            y (np.ndarray): probabilities of concepts indicating their likelihood whether they are visible (shape t, c)
            responses (np.ndarray): response (yes/no) of concepts whether they are visible (shape t, c)
            time (list): t time values (length t)
            delta_time (float): time duration for which a certain value is created (equals 1 / fps)
            config (dict): model config
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": probs,
        "responses": responses,
        "time": times,
        "delta_time": 1 / args.fps,
        "config": config,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--video", type=str, required=True, help="input video to process")
    parser.add_argument("-q", "--queries", type=str, nargs="+", required=True, help="path to query csv files")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file")

    # optional model parameters
    parser.add_argument("--model", type=str, default="Salesforce/instructblip-flan-t5-xl")
    parser.add_argument("--processor", type=str, default="Salesforce/instructblip-flan-t5-xl")

    # optional input parameters
    parser.add_argument("--fps", type=int, required=False, default=2, help="fps to process video")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=1920, help="max dimension of the video frames"
    )
    parser.add_argument("--debug", action="store_true", help="debug output")
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

    # load model
    model = BLIP2(model=args.model, processor=args.processor)

    # TODq load queries
    tasks = {}
    for csvfile in args.queries:
        task = os.path.splitext(os.path.basename(csvfile))[0]
        tasks[task] = []

        with open(csvfile) as f:
            spamreader = csv.reader(f, delimiter=",")
            for row in spamreader:
                tasks[task].append({"class": row[0], "query": row[1]})

    # decode video
    vd = VideoDecoder(path=args.video, max_dimension=args.max_dimension, fps=args.fps)

    times = []
    responses = {}
    for frame in vd:
        for task in tasks:
            if task not in responses:
                responses[task] = {}

            for query in tasks[task]:
                response = model.get_response(image=frame["frame"], query=query["query"])
                if query["class"] not in responses[task]:
                    responses[task][query["class"]] = []

                response = 1 if response == "yes" else 0
                responses[task][query["class"]].append(response)

                # TODO get probability of the response

        times.append(frame["time"])
        logging.debug(f"{frame['time']} s: {response}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"responses": responses, "times": times}, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
