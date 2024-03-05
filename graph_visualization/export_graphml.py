import argparse
import json
import logging
import math
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
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="path to .pkl folder"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yml",
        help="path to .yml config file",
    )
    parser.add_argument(
        "--asr_type", "-a", type=str, default="whisper", help="whisper | whisperx"
    )

    # output
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="path to .graphml folder"
    )

    # optional params
    parser.add_argument("--pyviz", action="store_true", help="debug output")
    parser.add_argument(
        "--speaker_seg_tol",
        type=float,
        default=3.0,
        help="tolerance [s] to combine speaker segments",
    )
    parser.add_argument(
        "--span_tol", type=float, default=3.0, help="tolerance [s] to combine spans"
    )
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
    fname = os.path.join(args.input, "asr_%s.pkl" % (args.asr_type))
    assert os.path.isfile(fname), f"Cannot find {fname}"

    data = read_pkl(fname)
    speaker_segments = data["output_data"]["speaker_segments"]

    # create speaker turns
    speaker_turns = get_speaker_turns(
        speaker_segments, gap=0.01
    )

    # add nodes for speaker turs
    G = add_speakerturn_nodes(G, speaker_turns, config["base"]["asr"])

    # add spans based on speaker turns and shots (smallest temporal element in our graph)
    G, spans = add_span_nodes(G, speaker_turns, shots, config, args.span_tol)

    # add edges between shots and speaker turns to spans
    G = add_edges_to_spans(G, shots, speaker_turns, spans)

    logging.info(f"# Graph created!")
    logging.info(f"   - Number of shots: {len(shots)}")
    logging.info(f"   - Number of speaker turns: {len(speaker_turns)}")
    logging.info(f"   - Number of spans: {len(spans)}")

    return G


def add_attributes_shot(G, plugin, data, config, args):
    logging.debug(f"Add attributes from {plugin} to shots")

    if "camera_size" in plugin:
        t_iter = enumerate(iter(data["output_data"]["time"]))
        idx, t = next(t_iter)

        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "shot":
                continue

            st = node_attr["start_time"]
            et = node_attr["end_time"]

            shot_idx = []
            while t <= st:
                try:
                    idx, t = next(t_iter)
                except StopIteration:
                    break

            while t <= et:
                if t >= data["output_data"]["time"][-1]:
                    break

                shot_idx.append(idx)
                try:
                    idx, t = next(t_iter)
                except StopIteration:
                    break

            if len(shot_idx) < 1:
                continue

            # get corresponding values
            shot_values = data["output_data"]["y"][
                shot_idx[0] : shot_idx[-1] + 1, :
            ]  # times x concepts

            # aggregate values for each concept
            # TODO: other aggregation functions
            mean_scores = list(np.mean(shot_values, axis=0))
            median_scores = list(np.median(shot_values, axis=0))

            if config["attribute"]["value"] == "median":
                scores = median_scores
            else:
                scores = mean_scores

            # add edges based on threshold
            for i, concept in enumerate(data["output_data"]["index"]):
                node_attr[f"{plugin}/{str(i)}/{concept}"] = scores[i]

            prediction = np.argmax(scores)
            node_attr[f"{plugin}/prediction"] = int(prediction)
            prediction_label = data["output_data"]["index"][prediction]
            node_attr["title"] += f", Shot Size: {prediction_label}"
    elif "audioClf" in plugin:
        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "shot":
                continue

            # TODO Gullal: Why can it happen that the audioClf output has less entries as there are shots?
            if node_attr["index"] > len(data["output_data"]) - 1:
                logging.warning("audioClf: Number of shots does not match output data")
                continue

            node_attr["title"] += "\n"
            ## To handle None values added for cuts with no time difference
            if "None" in data["output_data"][node_attr["index"]]["top3_label"]:
                node_attr[f"{plugin}/None"] = 1
                node_attr["title"] += f"None: 1, "
            else:
                for lab, prob in zip(
                    data["output_data"][node_attr["index"]]["top3_label"],
                    data["output_data"][node_attr["index"]]["top3_label_prob"],
                ):
                    node_attr[f"{plugin}/{lab}"] = round(prob, 2)
                    node_attr["title"] += f"{lab}: {round(prob,2)}, "
    elif "face_analysis" in plugin:
        # get plugin fps and face information
        fps = data["plugins"][0]["parameters"]["fps"]
        faces = data["faces"]

        # create node for face clusters
        for face in faces:
            cluster_id = face["cluster_id"]
            if not G.has_node(cluster_id):
                G.add_node(
                    f"person_{cluster_id}",
                    label=f"person_{cluster_id}",
                    title=f"person_{cluster_id}",
                    type="person",
                )

        # loop over shots
        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "shot":
                continue

            # get index, start time, end time of the shot
            st = node_attr["start_time"]
            et = node_attr["end_time"]
            shot_idx = node_attr["index"]

            # calculate the number of frames the shot contains for normalization
            num_frames = max(1, math.floor((et - st) * fps))

            # iterate over faces and add attributes to the shot node if the face is within the shot
            persons_shot = {}
            for face in faces:
                if face["time"] < st or face["time"] > et:
                    continue

                # get cluster id, emotion, and headpose of the face
                cluster_id = face["cluster_id"]
                emotion = face["emotion"]
                headpose = face["headpose"]

                # check if face is an actor
                # TODO: better implementation
                is_actor = face["bbox"]["h"] > 0.1

                # init data for persons in shot
                if cluster_id not in persons_shot:
                    persons_shot[cluster_id] = {
                        # information for faces
                        "face": {
                            "cnt": 0,  # counter for occurrences in the whole frame
                            "cnt_pos": [0, 0, 0],  # ... left, right, center
                            "emotion": np.zeros(emotion.shape),  # 7 basic emotions
                            "headpose": np.zeros(headpose.shape),  # yaw, pitch, roll
                        },
                        # information for actors
                        "actor": {
                            "cnt": 0,  # counter for occurrences in the whole frame
                            "cnt_pos": [0, 0, 0],  # ... left, right, center
                            "emotion": np.zeros(emotion.shape),  # 7 basic emotions
                            "headpose": np.zeros(headpose.shape),  # yaw, pitch, roll
                        },
                    }

                # count number of frames the person is visible
                persons_shot[cluster_id]["face"]["cnt"] += 1
                persons_shot[cluster_id]["face"]["emotion"] += emotion
                persons_shot[cluster_id]["face"]["headpose"] += headpose
                if is_actor:
                    persons_shot[cluster_id]["actor"]["cnt"] += 1
                    persons_shot[cluster_id]["actor"]["emotion"] += emotion
                    persons_shot[cluster_id]["actor"]["headpose"] += headpose

                # Compute centre of the face bbox
                mid_x = face["bbox"]["x"] + face["bbox"]["w"] / 2

                # Compute the relative position of the face bbox
                # Count faces per position in shot
                idx = min(math.floor(mid_x * 3), 2)
                persons_shot[cluster_id]["face"]["cnt_pos"][idx] += 1
                if is_actor:
                    persons_shot[cluster_id]["actor"]["cnt_pos"][idx] += 1

            # get statistics over all faces in the shot
            # add edges for visible person (i.e., face clusters) to the shot
            shot_cnt = {
                "face": {"cnt": 0, "cnt_pos": [0, 0, 0]},
                "actor": {"cnt": 0, "cnt_pos": [0, 0, 0]},
            }

            for person_id, data in persons_shot.items():
                # print(data)
                for t in ["face", "actor"]:
                    # statistics over all faces
                    shot_cnt[t]["cnt"] += data[t]["cnt"] / num_frames

                    for i in range(len(shot_cnt["face"]["cnt_pos"])):
                        shot_cnt[t]["cnt_pos"][i] += data[t]["cnt_pos"][i] / num_frames

                    if data[t]["cnt"] < 1:
                        continue

                    # add relation of person to shot
                    G.add_edge(
                        # edge (from, to)
                        f"person_{person_id}",
                        f"shot_{shot_idx}",
                        # face counts are divided by the number of frames
                        # to get the percentage of visibility within the shot
                        weight=data[t]["cnt"] / num_frames,
                        face_count=data[t]["cnt"] / num_frames,
                        face_count_left=data[t]["cnt_pos"][0] / num_frames,
                        face_count_center=data[t]["cnt_pos"][1] / num_frames,
                        face_count_right=data[t]["cnt_pos"][2] / num_frames,
                        # attributes are divided by the number of times the face
                        # is visible to get an average value within the shot
                        face_emotion_angry=data[t]["emotion"][0] / data[t]["cnt"],
                        face_emotion_disgust=data[t]["emotion"][1] / data[t]["cnt"],
                        face_emotion_fear=data[t]["emotion"][2] / data[t]["cnt"],
                        face_emotion_happy=data[t]["emotion"][3] / data[t]["cnt"],
                        face_emotion_sad=data[t]["emotion"][4] / data[t]["cnt"],
                        face_emotion_surprise=data[t]["emotion"][5] / data[t]["cnt"],
                        face_emotion_neutral=data[t]["emotion"][6] / data[t]["cnt"],
                        #
                        face_headpose_yaw=data[t]["headpose"][0] / data[t]["cnt"],
                        face_headpose_pitch=data[t]["headpose"][1] / data[t]["cnt"],
                        face_headpose_roll=data[t]["headpose"][2] / data[t]["cnt"],
                    )

            # face count in the shot
            node_attr[f"{plugin}/face_count"] = shot_cnt["face"]["cnt"]
            # relative position of the faces in the shot
            node_attr[f"{plugin}/face_count_left"] = shot_cnt["face"]["cnt_pos"][0]
            node_attr[f"{plugin}/face_count_center"] = shot_cnt["face"]["cnt_pos"][1]
            node_attr[f"{plugin}/face_count_right"] = shot_cnt["face"]["cnt_pos"][2]

            # actor count in the shot
            node_attr[f"{plugin}/actor_count"] = shot_cnt["actor"]["cnt"]
            # relative position of the actors in the shot
            node_attr[f"{plugin}/actor_count_left"] = shot_cnt["actor"]["cnt_pos"][0]
            node_attr[f"{plugin}/actor_count_center"] = shot_cnt["actor"]["cnt_pos"][1]
            node_attr[f"{plugin}/actor_count_right"] = shot_cnt["actor"]["cnt_pos"][2]

            node_attr[
                "title"
            ] += f"\nFace Count: {shot_cnt['face']['cnt']}, Face Positions: {shot_cnt['face']['cnt_pos']}"

    return G


def add_attributes_speakerturn(G, plugin, data, config):
    logging.debug(f"Add attributes from {plugin} to speaker turns")

    if "sentiment" in plugin:
        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "speaker_turn":
                continue

            # TODO Gullal: Why can it happen that the sentiment output has less entries as there are speaker turns?
            if node_attr["index"] > len(data["output_data"]["sentence_wise"]) - 1:
                logging.warning(
                    "sentiment: Number of speaker turns does not match output data"
                )
                continue

            vector = data["output_data"]["sentence_wise"][node_attr["index"]]["vector"]
            proportion_sentiment = config["labels"][np.argmax(vector)]
            proportion_max_value = round(np.max(vector), 2)
            prediction = data["output_data"]["speakerturn_wise"][node_attr["index"]][
                "pred"
            ]
            max_value = round(
                np.max(
                    data["output_data"]["speakerturn_wise"][node_attr["index"]]["prob"]
                ),
                2,
            )
            node_attr[f"{plugin}/positive_ratio"] = round(vector[0], 2)
            node_attr[f"{plugin}/negative_ratio"] = round(vector[1], 2)
            node_attr[f"{plugin}/neutral_ratio"] = round(vector[2], 2)
            node_attr[f"{plugin}/prediction"] = prediction
            # Change Speaker Turn node color based on sentiment: green - positive, red - negative, light blue - neutral
            if prediction == "positive":
                node_attr["color"] = "#00ff00"
            elif prediction == "negative":
                node_attr["color"] = "#ff0000"
            else:
                node_attr["color"] = "#00ffff"

            node_attr[
                "title"
            ] += f"\nOverall Sentiment: {prediction} [{max_value}], \
                                    Proportion Sentiment: {proportion_sentiment} [{proportion_max_value}]"
    elif "pos" in plugin:
        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "speaker_turn":
                continue

            pos_vector = data["output_data"]["speakerturn_wise"][node_attr["index"]][
                "vector"
            ]

            node_attr["title"] += "\n"
            cnt = 0
            for ind, key in config["labels"].items():
                node_attr[f"{plugin}/{key}/count"] = pos_vector[ind]

                # TODO Gullal:
                # After fixing sentiment and audiClf, I received a warning here caused by a division with 0
                # So, I guess the reason is that you somehow did not store outputs for sentiment and audioClf
                # if there was a speaker turn with 0 words
                if node_attr["num_words"] > 0:
                    node_attr[f"{plugin}/{key}/frequency"] = (
                        pos_vector[ind] / node_attr["num_words"]
                    )
                else:
                    logging.warning("pos: Speaker turn with 0 words")
                    node_attr[f"{plugin}/{key}/frequency"] = 0

                if cnt == 6:
                    node_attr["title"] += "\n"
                node_attr["title"] += f"{key}: {pos_vector[ind]}, "
                cnt += 1
    elif "audioClf" in plugin:
        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "speaker_turn":
                continue

            # TODO Gullal: Why can it happen that the audioClf output has less entries as there are speaker turns?
            if node_attr["index"] > len(data["output_data"]) - 1:
                logging.warning(
                    "audioClf: Number of speaker turns does not match output data"
                )
                continue

            node_attr["title"] += "\n"
            ## To handle None values added for speaker turn with 0 words
            if "None" in data["output_data"][node_attr["index"]]["top3_label"]:
                node_attr[f"{plugin}/None"] = 1
                node_attr["title"] += f"None: 1, "
            else:
                for lab, prob in zip(
                    data["output_data"][node_attr["index"]]["top3_label"],
                    data["output_data"][node_attr["index"]]["top3_label_prob"],
                ):
                    node_attr[f"{plugin}/{lab}"] = round(prob, 2)
                    node_attr["title"] += f"{lab}: {round(prob,2)}, "
    elif "segmentClf" in plugin:
        for node in G.nodes(data=True):
            _, node_attr = node
            if node_attr["type"] != "speaker_turn":
                continue

            gender = data["output_data"][node_attr["index"]]["gender_pred"]
            gender_prob = round(
                float(data["output_data"][node_attr["index"]]["gender_prob"]), 2
            )
            node_attr[f"{plugin}/gender/{gender}"] = gender_prob
            node_attr["title"] += f"\n{gender}: {gender_prob}"

            ## Check if object is not list
            if not isinstance(data["output_data"][node_attr["index"]]["emotion_pred_top3"], list):
                node_attr[f"{plugin}/{data['output_data'][node_attr['index']]['emotion_pred_top3']}"] = round(
                    float(data["output_data"][node_attr["index"]]["emotion_prob_top3"]), 2
                )
                node_attr["title"] += f", {data['output_data'][node_attr['index']]['emotion_pred_top3']}: \
                    {round(float(data['output_data'][node_attr['index']]['emotion_prob_top3']), 2)}, "
            else:
                for lab, prob in zip(
                        data["output_data"][node_attr["index"]]["emotion_pred_top3"],
                        data["output_data"][node_attr["index"]]["emotion_prob_top3"],
                    ):
                        node_attr[f"{plugin}/{lab}"] = round(prob, 2)
                        node_attr["title"] += f", {lab}: {round(prob,2)}, "
    return G


def add_ner_nodes(G, plugin, data, config):
    logging.debug(f"Add nodes and relations for {plugin}")

    for i, segment in enumerate(data["output_data"]["speakerturn_wise"]):
        entities = segment["tags"]
        for ent in entities:
            if not G.has_node(ent["wd_label"]):
                title = "WD-Label: %s, Url: %s" % (ent["wd_label"], ent["url"])
                label = ent["text"] if ent["wd_label"] == "unk" else ent["wd_label"]
                if "PER" in ent["type"]:
                    G.add_node(
                        label, label=label, title=title, type=plugin, color="pink"
                    )
                elif "ORG" in ent["type"]:
                    G.add_node(
                        label, label=label, title=title, type=plugin, color="yellow"
                    )
                elif "LOC" in ent["type"]:
                    G.add_node(
                        label, label=label, title=title, type=plugin, color="purple"
                    )
                elif "EVENT" in ent["type"]:
                    G.add_node(
                        label, label=label, title=title, type=plugin, color="brown"
                    )
                else:
                    G.add_node(
                        label, label=label, title=title, type=plugin, color="gray"
                    )

                G.add_edge(
                    f"spk_turn_{i}", label, title="mentioned as %s" % (ent["text"])
                )

    return G


def add_nodes(G, plugin, data, config):
    if "ner" in plugin:
        G = add_ner_nodes(G, plugin, data, config)
    else:
        add_node_function = {
            "query": add_query_nodes,
        }

        add_relation_function = {
            "shot": add_shot_relation,
        }

        G = add_node_function[config["plugin_type"]](
            G=G, plugin=plugin, data=data, config=config
        )
        G = add_relation_function[config["relation"]["target"]](
            G=G, plugin=plugin, data=data, config=config
        )
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
        shot_values = data["responses"][
            :, span_places[0] : span_places[-1] + 1
        ]  # concepts x times

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
                G.add_edge(
                    concept,
                    f"shot_{idx}",
                    weight=scores[i],
                    mean=mean_scores[i],
                    median=median_scores[i],
                )

    return G


def add_shot_nodes(G, data, config):
    for i, shot in enumerate(data):
        start_time = round(shot["start"], 2)
        end_time = round(shot["end"], 2)

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
        start_time = round(turn["start"], 2)
        end_time = round(turn["end"], 2)

        speaker_label = turn["speaker"] if turn["speaker"] else "None"
        num_words = turn["n_words"]

        # Create the label and title for the node (to see the attribute values and label nodes)
        label = f"Spk. turn {i}"
        title = f"Speaker ID: {speaker_label}, Num Words: {num_words}, Start: {start_time}, End: {end_time}"

        G.add_node(
            f"spk_turn_{i}",
            label=label,
            title=title,
            index=i,
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


# def get_speaker_turns(speaker_segments, speaker_seg_tol=3.0):
#     for i, segment in enumerate(speaker_segments):
#         start_time = round(segment["start"], 2)
#         end_time = round(segment["end"], 2)
#         if i < len(speaker_segments) - 1:
#             if speaker_segments[i + 1]["start"] - end_time < speaker_seg_tol:
#                 end_time = round(
#                     speaker_segments[i + 1]["start"] - 0.04, 2
#                 )  # Similar to shots

#         segment["start"] = start_time
#         segment["end"] = end_time
#         segment["n_words"] = len(segment["text"].split())

#     speaker_turns = []
#     current_span = None
#     for segment in speaker_segments:
#         if current_span is None or segment["speaker"] != current_span["speaker"]:
#             current_span = segment.copy()
#             speaker_turns.append(current_span)
#         else:
#             current_span["end"] = segment["end"]
#             current_span["text"] += " " + segment["text"].strip()
#             current_span["n_words"] += segment["n_words"]

#     return speaker_turns, speaker_segments

def get_speaker_turns(speaker_segments, gap=0.01):
    speaker_segments = sorted(speaker_segments, key=lambda x: x["start"])

    speaker_turns = []

    for segment in speaker_segments:

        segment["n_words"] = len(segment["text"].split()) if "text" in segment else 0

        if "speaker" not in segment or segment["speaker"] is None:
            segment["speaker"] = "Unknown"

        if not speaker_turns:
            speaker_turns.append(segment.copy())
        else:
            last_turn = speaker_turns[-1]

            ## Checking if current segment belongs to same speaker
            if last_turn["speaker"] == segment["speaker"]:
                # last_turn["end"] = max(last_turn["end"], segment["end"]) # If no gap to be add
                last_turn["end"] = max(last_turn["end"], segment["end"] - gap)   # Just to ensure some gap between turns

                last_turn["text"] += " " + segment["text"].strip()
                last_turn["n_words"] += len(segment["text"].split())
            else:
            ## Treat otherwise as a new speaker turn
                if segment["start"] - last_turn["end"] < gap:       ## Comment if no gap to add
                    segment["start"] = last_turn["end"] + gap
                
                speaker_turns.append(segment.copy())
    
    return speaker_turns


def add_span_nodes(G, speaker_turns, shots, config, tolerance=3.0):
    # Combine the two lists and sort by start time
    intervals = sorted(
        [{"start": shot["start"], "end": shot["end"]} for shot in shots]
        + [
            {"start": speaker["start"], "end": speaker["end"]}
            for speaker in speaker_turns
        ],
        key=lambda interval: interval["end"],
    )

    spans = []
    current_end = 0.0

    for i, interval in enumerate(intervals):
        start_time = interval["start"]
        end_time = interval["end"]

        if start_time >= current_end and (start_time - current_end) < tolerance:
            # For intervals where start time is more than current_end and under tolerance
            span_start = current_end
            span_end = end_time
            if (
                i < len(intervals) - 1
                and abs(intervals[i + 1]["start"] - end_time) < tolerance
            ):
                span_end = intervals[i + 1]["start"]

            spans.append({"start": span_start, "end": span_end})
            current_end = span_end
        elif start_time >= current_end and (start_time - current_end) > tolerance:
            # Add filler span first if gap is greater than or equal to 3 seconds
            span_start = current_end
            span_end = start_time
            spans.append(
                {"start": span_start, "end": span_end}
            )  # Add the filler span first

            span_start = start_time
            span_end = end_time
            spans.append(
                {"start": span_start, "end": span_end}
            )  # Add current interval now

            current_end = span_end
        else:
            # For intervals end time is greater than current (to handle shot intervals shorter than speaker turns)
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
        # Strictly between the spans
        #         span_events = list(set([e for e, t in event_list if t > start_time and t < end_time]))

        #         for e in span_events:
        #             G.add_edge(e, f"span_{i}")

        span_places = list(
            set([p for p, t in place_list if t > start_time and t < end_time])
        )

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
            f"span_{j}"
            for j, span in enumerate(spans)
            if not (end_time <= span["start"] or start_time >= span["end"])
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
            f"span_{j}"
            for j, span in enumerate(spans)
            if not (end_time <= span["start"] or start_time >= span["end"])
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

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    # read config
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    logging.debug(config)

    # create base structure of the graph
    G = create_base_graph(args, config)

    # add additional nodes to the graph
    for plugin in config["nodes"]:
        if plugin in [
            "transnet_shotdetection",
            "asr",
            "span",
        ]:  # already computed for base graph
            continue

        fname = os.path.join(args.input, plugin + ".pkl")
        if args.asr_type == "whisperx":
            fname = fname.replace("whisper", "whisperx")
        assert os.path.isfile(fname), f"Cannot find: {fname}"

        data = read_pkl(fname)

        if config["nodes"][plugin]["type"] == "node":
            G = add_nodes(G, plugin, data, config["nodes"][plugin])
        else:  # attribute
            if config["nodes"][plugin]["attribute"]["target"] == "shot":
                G = add_attributes_shot(G, plugin, data, config["nodes"][plugin], args)
            elif config["nodes"][plugin]["attribute"]["target"] == "speaker":
                add_attributes_speakerturn(G, plugin, data, config["nodes"][plugin])

    print(G)

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
            json.dumps(
                {
                    "physics": {
                        "solver": "barnesHut",
                        "hierarchicalRepulsion": {"nodeDistance": 150},
                    }
                }
            )
        )
        outfile = os.path.join(args.output, f"{vidname}.html")

        logging.info(f"Store pyviz network to : {outfile}")
        net.save_graph(os.path.join(args.output, f"{vidname}.html"))


if __name__ == "__main__":
    sys.exit(main())
