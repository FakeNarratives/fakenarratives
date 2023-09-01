import argparse
import json
import logging
import networkx as nx
import numpy as np
import os
from pyvis.network import Network
import sys
import yaml

from utils import read_pkl


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")

    # inputs
    parser.add_argument("--input", "-i", type=str, required=True, help="path to .pkl folder")
    parser.add_argument("--config", "-c", type=str, default="config.yml", help="path to .yml config file")

    # output
    parser.add_argument("--output", "-o", type=str, required=True, help="path to .graphml folder")

    # optional params
    parser.add_argument("--pyviz", action="store_true", help="debug output")
    parser.add_argument("--speaker_seg_tol", type=float, default=3.0, help="tolerance [s] to combine speaker segments")
    parser.add_argument("--span_tol", type=float, default=3.0, help="tolerance [s] to combine spans")
    args = parser.parse_args()
    return args


def create_base_graph(args, config):
    G = nx.Graph()

    # read shots
    logging.info(f"Create shots")
    fname = os.path.join(args.input, "transnet_shotdetection.pkl")
    assert os.path.isfile(fname), f"Cannot find {fname}"

    data = read_pkl(fname)
    shots = data["output_data"]["shots"]

    # add nodes for shots
    G = add_shot_nodes(G, shots, config["base"]["transnet_shotdetection"])

    # read speaker segments
    logging.info(f"Create speaker segments")
    fname = os.path.join(args.input, "asr.pkl")
    assert os.path.isfile(fname), f"Cannot find {fname}"

    data = read_pkl(fname)
    speaker_segments = data["output_data"]["speaker_segments"]

    # create speaker turns
    speaker_turns, speaker_segments = get_speaker_turns(speaker_segments, args.speaker_seg_tol)

    # add nodes for speaker turs
    G = add_speakerturn_nodes(G, speaker_turns, config["base"]["asr"])

    # add spans based on speaker turns and shots (smallest temporal element in our graph)
    G, spans = add_span_nodes(G, speaker_turns, shots, config, args.span_tol)

    # add edges between shots and speaker turns to spans
    G = add_edges_to_spans(G, shots, speaker_turns, spans)

    logging.info(f"## Graph created!")
    logging.info(f"   - Number of shots: {len(shots)}")
    logging.info(f"   - Number of speaker turns: {len(speaker_turns)}")
    logging.info(f"   - Number of spans: {len(spans)}")

    return G


def add_nodes(G, plugin, data, config):
    add_node_function = {
        "query": add_query_nodes,
    }

    add_relation_function = {
        "shot": add_shot_relation,
    }

    G = add_node_function[config["plugin_type"]](G=G, plugin=plugin, data=data, config=config)
    G = add_relation_function[config["relation"]["target"]](G=G, plugin=plugin, data=data, config=config)
    return G


def add_query_nodes(G, plugin, data, config):
    logging.debug(f"Add nodes for {plugin}")
    for label in data["labels"]:
        G.add_node(
            label,
            label=label,
            type=plugin,
            color=config["color"],
            shape=config["shape"],
        )
    return G


def add_shot_relation(G, plugin, data, config):
    logging.debug(f"Add relations from {plugin} to {config['relation']}")

    for node in G.nodes(data=True):
        _, node_attr = node
        if node_attr["type"] != "shot":
            continue

        # get shot boundaries
        st = node_attr["start_time"]
        et = node_attr["end_time"]
        idx = node_attr["index"]

        # assign timestamps of the data to the shot boundaries
        span_places = [i for i, t in enumerate(data["time"]) if t >= st and t <= et]
        if len(span_places) < 1:
            continue

        # get corresponding values
        shot_values = data["responses"][:, span_places[0] : span_places[-1] + 1]  # concepts x times

        # aggregate values for each concept
        # TODO: other aggregation functions
        mean_scores = list(np.mean(shot_values, axis=-1))
        median_scores = list(np.median(shot_values, axis=-1))

        if config["relation"]["weight"] == "median":
            scores = median_scores
        else:
            scores = mean_scores

        # add edges based on threshold
        for i, concept in enumerate(data["labels"]):
            if scores[i] > config["relation"]["threshold"]:
                G.add_edge(concept, f"shot_{idx}", weight=scores[i], mean=mean_scores[i], median=median_scores[i])

    return G


def add_shot_nodes(G, data, config):
    for i, shot in enumerate(data):
        start_time = shot["start"]
        end_time = shot["end"]

        label = f"Shot {i}"
        title = f"Start: {start_time}, End: {end_time}"

        G.add_node(
            f"shot_{i}",
            label=label,
            title=title,
            index=i,
            type="shot",
            start_time=start_time,
            end_time=end_time,
            color=config["color"],
            shape=config["shape"],
        )

    # add edges between shots
    for i in range(len(data) - 1):
        G.add_edge(f"shot_{i}", f"shot_{i+1}")

    return G


def add_speakerturn_nodes(G, speaker_turns, config):
    for i, turn in enumerate(speaker_turns):
        start_time = turn["start"]
        end_time = turn["end"]

        speaker_label = turn["speaker"] if turn["speaker"] else "None"
        num_words = turn["n_words"]

        # Create the label and title for the node (to see the attribute values and label nodes)
        label = f"Spk. turn {i}"
        title = f"Speaker ID: {speaker_label}, Num Words: {num_words}, Start: {start_time}, End: {end_time}"

        G.add_node(
            f"spk_turn_{i}",
            label=label,
            title=title,
            type="speaker_turn",
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_label,
            num_words=num_words,
            color=config["color"],
            shape=config["shape"],
        )

    # add edges between speaker turns
    for i in range(len(speaker_turns) - 1):
        G.add_edge(f"spk_turn_{i}", f"spk_turn_{i+1}")

    return G


def get_speaker_turns(speaker_segments, speaker_seg_tol=3.0):
    for i, segment in enumerate(speaker_segments):
        start_time = round(segment["start"], 2)
        end_time = round(segment["end"], 2)
        if i < len(speaker_segments) - 1:
            if speaker_segments[i + 1]["start"] - end_time < speaker_seg_tol:
                end_time = round(speaker_segments[i + 1]["start"] - 0.04, 2)  ## Similar to shots

        segment["start"] = start_time
        segment["end"] = end_time
        segment["n_words"] = len(segment["text"].split())

    speaker_turns = []
    current_span = None
    for segment in speaker_segments:
        if current_span is None or segment["speaker"] != current_span["speaker"]:
            current_span = segment.copy()
            speaker_turns.append(current_span)
        else:
            current_span["end"] = segment["end"]
            current_span["text"] += " " + segment["text"].strip()
            current_span["n_words"] += segment["n_words"]

    return speaker_turns, speaker_segments


def add_span_nodes(G, speaker_turns, shots, config, tolerance=3.0):
    # Combine the two lists and sort by start time
    intervals = sorted(
        [{"start": shot["start"], "end": shot["end"]} for shot in shots]
        + [{"start": speaker["start"], "end": speaker["end"]} for speaker in speaker_turns],
        key=lambda interval: interval["end"],
    )

    spans = []
    current_end = 0.0

    for i, interval in enumerate(intervals):
        start_time = interval["start"]
        end_time = interval["end"]

        if start_time >= current_end and (start_time - current_end) < tolerance:
            ## For intervals where start time is more than current_end and under tolerance
            span_start = current_end
            span_end = end_time
            if i < len(intervals) - 1 and abs(intervals[i + 1]["start"] - end_time) < tolerance:
                span_end = intervals[i + 1]["start"]

            spans.append({"start": span_start, "end": span_end})
            current_end = span_end
        elif start_time >= current_end and (start_time - current_end) > tolerance:
            ## Add filler span first if gap is greater than or equal to 3 seconds
            span_start = current_end
            span_end = start_time
            spans.append({"start": span_start, "end": span_end})  ## Add the filler span first

            span_start = start_time
            span_end = end_time
            spans.append({"start": span_start, "end": span_end})  ## Add current interval now

            current_end = span_end
        else:
            ## For intervals ent time is greater than current (to handle shot intervals shorter than speaker turns)
            if end_time > current_end:
                spans.append({"start": current_end, "end": end_time})
                current_end = end_time

    for i, span in enumerate(spans):
        start_time = span["start"]
        end_time = span["end"]

        label = f"Span {i}"
        title = f"Start: {start_time}, End: {end_time}"

        G.add_node(
            f"span_{i}",
            label=label,
            title=title,
            type="span",
            start_time=start_time,
            end_time=end_time,
            color=config["base"]["span"]["color"],
        )

    for i in range(len(spans) - 1):
        G.add_edge(f"span_{i}", f"span_{i+1}")

    return G, spans


def link_attributes_to_spans(G, spans, event_list, place_list):
    for i, span in enumerate(spans):
        start_time = span["start"]
        end_time = span["end"]
        ## Strictly between the spans
        #         span_events = list(set([e for e, t in event_list if t > start_time and t < end_time]))

        #         for e in span_events:
        #             G.add_edge(e, f"span_{i}")

        span_places = list(set([p for p, t in place_list if t > start_time and t < end_time]))

        for p in span_places:
            G.add_edge(p, f"span_{i}")

    return G


def add_edges_to_spans(G, shots, speaker_turns, spans):
    # Iterate over the shots
    for i, shot in enumerate(shots):
        start_time = shot["start"]
        end_time = shot["end"]

        # Find the spans that overlap with this shot
        overlapping_spans = [
            f"span_{j}" for j, span in enumerate(spans) if not (end_time <= span["start"] or start_time >= span["end"])
        ]

        # Add edges from the shot to the overlapping spans
        for span_id in overlapping_spans:
            G.add_edge(f"shot_{i}", span_id)

    # Iterate over the speaker turns
    for i, turn in enumerate(speaker_turns):
        start_time = turn["start"]
        end_time = turn["end"]

        # Find the spans that overlap with this speaker segment
        overlapping_spans = [
            f"span_{j}" for j, span in enumerate(spans) if not (end_time <= span["start"] or start_time >= span["end"])
        ]

        # Add edges from the speaker segment to the overlapping spans
        for span_id in overlapping_spans:
            G.add_edge(f"spk_turn_{i}", span_id)

    return G


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # read config
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    logging.debug(config)

    # create base structure of the graph
    G = create_base_graph(args, config)

    # add additional nodes to the graph
    for plugin in config["nodes"]:
        if plugin in ["transnet_shotdetection", "asr", "span"]:  # already computed for base graph
            continue

        fname = os.path.join(args.input, plugin + ".pkl")
        assert os.path.isfile(fname), f"Cannot find: {fname}"

        data = read_pkl(fname)
        G = add_nodes(G, plugin, data, config["nodes"][plugin])

    # export graph
    os.makedirs(args.output, exist_ok=True)
    vidname = os.path.basename(args.input)
    outfile = os.path.join(args.output, f"{vidname}.graphml")

    logging.info(f"Store networkx graph to: {outfile}")
    nx.write_graphml(G, outfile)

    # optional: create pyviz graph
    if args.pyviz:
        net = Network()
        net.from_nx(G)
        net.set_options(
            json.dumps({"physics": {"solver": "barnesHut", "hierarchicalRepulsion": {"nodeDistance": 150}}})
        )
        outfile = os.path.join(args.output, f"{vidname}.html")

        logging.info(f"Store pyviz network to : {outfile}")
        net.save_graph(os.path.join(args.output, f"{vidname}.html"))


if __name__ == "__main__":
    sys.exit(main())
