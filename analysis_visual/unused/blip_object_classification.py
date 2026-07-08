import argparse
import csv
import logging
import numpy as np
import os
import pickle
import sys
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

sys.path.append("..")
from project_utils import read_video_and_get_info, set_seeds


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
        precision=4
    ):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if precision == 4:
            self._dtype = torch.bfloat16
            self._model = InstructBlipForConditionalGeneration.from_pretrained(
                model,
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=self._dtype,
            )
            
        elif precision == 8:
            self._dtype = torch.bfloat16
            self._model = InstructBlipForConditionalGeneration.from_pretrained(
                model,
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=self._dtype,
            )
        else:  # 32bit
            self._model = InstructBlipForConditionalGeneration.from_pretrained(model)
            self._model.to(self._device)
            
        self._processor = InstructBlipProcessor.from_pretrained(processor)

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
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(self._device, dtype=self._dtype)
        outputs = self._model.generate(**inputs, **self._params)
        return self._processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def instructblip_classification_pkl(
    probs: list,
    responses: list,
    times: list,
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

    labels = []
    outputs = []
    for label, response in responses.items():
        labels.append(label)

        label_outputs = []
        for r in response:
            if r == "yes":
                out = 1
            elif r == "no":
                out = 0
            else:  # what to do if InstructBlip decides to not provide yes/no as an answer?
                out = 0
                # logging.warning(f"Answer {r}: set to 0")

            label_outputs.append(out)
        
        outputs.append(label_outputs)
    
    outputs = np.asarray(outputs)
    logging.debug(labels)
    logging.debug(outputs.shape)

    return {
        "labels": labels,
        # "y": probs,
        "responses": outputs,
        "time": times,
        "delta_time": 1 / args.fps,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-q", "--queries", type=str, nargs="+", required=True, help="path to query csv files")
    parser.add_argument("-o", "--output", type=str, required=True, help="output file")

    # optional model parameters
    parser.add_argument("--model", type=str, default="Salesforce/instructblip-flan-t5-xl")
    parser.add_argument("--processor", type=str, default="Salesforce/instructblip-flan-t5-xl")
    parser.add_argument("--precision", type=int, default=4, help="weight precision of the model")

    # optional input parameters
    parser.add_argument("--fps", type=int, required=False, default=1, help="fps to process video")
    parser.add_argument(
        "--max_dimension", type=int, required=False, default=512, help="max dimension of the video frames"
    )
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers to use for data loading")
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    set_seeds(42)

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # load model
    # model = BLIP2(model=args.model, processor=args.processor)

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
    # loop trough input videos
    videos = args.videos
    for video_path in videos:
        # Load video
        vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, args.num_workers, 1)
        logging.info(f"\tVideo info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
        total_frames = len(vd)

        times = []
        responses = {}
        
        # for frame in vd:
        for frame_no in range(0, total_frames):
            for task in tasks:
                if task not in responses:
                    responses[task] = {}

                for query in tasks[task]:
                    # query_question = f"{query['query']} Please answer with yes or no!"
                    # response = model.get_response(image=frame["frame"], query=query_question)
                    if query["class"] not in responses[task]:
                        responses[task][query["class"]] = []

                    # logging.debug(f"{frame['time']} (task: {task}; query: {query['class']}): {response}")
                    # response = 1 if response == "yes" else 0  # QUESTION: should we do that?
                    # responses[task][query["class"]].append(response)
                    responses[task][query["class"]].append(0)  ## Dummy value for now

                    # TODO get probability of the response

            # times.append(frame["time"])
            times.append(frame_no)

        # setup output dir
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.output, vidname)
        os.makedirs(output_dir, exist_ok=True)

        for task in responses:
            output_dict = instructblip_classification_pkl(probs=[], responses=responses[task], times=times, args=args)

            output_file = os.path.join(output_dir, f"instructblip_{task}.pkl")
            with open(output_file, "wb") as f:
                pickle.dump(output_dict, f)

            logging.info(f"Output written to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
