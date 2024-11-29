import argparse
import logging
import os
import pickle
from pympi.Elan import Eaf, to_eaf
import sys

## Shots, Speaker Turns - Text, Topic - Name

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-i", "--input", type=str, help="input folder")
    parser.add_argument("-o", "--output", type=str, help="output folder")
    parser.add_argument("-c", "--channel", type=str, help="channel name")

    parser.add_argument("--shots", action="store_true", help="save shots into eaf")
    parser.add_argument("--turns", action="store_true", help="save trun text into eaf")
    parser.add_argument("--topic", action="store_true", help="save topic into eaf")
    parser.add_argument("--spk_attr", action="store_true", help="save turn attributes into eaf")
    parser.add_argument("--vlm", action="store_true", help="save vlm classifications into eaf")
    parser.add_argument("--audioclf", action="store_true", help="save audio classification into eaf")

    parser.add_argument("-vv", "--debug", action="store_true", help="debug output")
    args = parser.parse_args()
    return args


def create_elan_file(video_name):
    eaf = Eaf(author="")
    eaf.remove_tier("default")
    eaf.add_linked_file(file_path=f"{os.path.basename(video_name)}", mimetype="video/%s"%(video_name.split('.')[-1]))

    return eaf


def create_shot_tier(eaf, input_pickle):
    logging.info(f"Reading: {input_pickle}")

    # create tier
    tier = "shots"
    eaf.add_tier(tier_id=tier)

    # loop through shots and add annotations
    shots = pickle.load(open(input_pickle, "rb"))
    i = 0
    for shot in shots["output_data"]["shots"]:
        logging.debug(shot)

        start = int(1000 * shot["start"])
        end = int(1000 * shot["end"])

        if start == end:
            continue

        eaf.add_annotation(tier, start=start, end=end, value=str(i))
        i += 1

    return eaf


def create_turns_tier(eaf, input_pickle):
    logging.info(f"Reading: {input_pickle}")

    # create tier
    tier_speaker = "speaker_turn"
    eaf.add_tier(tier_id=tier_speaker)

    tier_transcript = "transcript"
    eaf.add_tier(tier_id=tier_transcript)

    # loop through shots and add annotations
    asr = pickle.load(open(input_pickle, "rb"))
    i = 0

    for speaker_segment in asr["output_data"]["speaker_turns"]:
        logging.debug(speaker_segment)

        start = int(1000 * speaker_segment["start"])
        end = int(1000 * speaker_segment["end"])

        if start >= end:
            continue

        eaf.add_annotation(
            tier_speaker, start=start, end=end, value=speaker_segment["speaker"].strip()
        )
        eaf.add_annotation(
            tier_transcript, start=start, end=end, value=speaker_segment["text"].strip()
        )
        i += 1

    return eaf


def create_topic_tier(eaf, input_pickle, asr_pickle):
    logging.info(f"Reading: {input_pickle}")

    topics = pickle.load(open(input_pickle, "rb"))
    asr = pickle.load(open(asr_pickle, "rb"))

    topic_dict = {topic["turn_index"]: topic for topic in topics}

    # create tier
    tier_speaker = "topic"
    eaf.add_tier(tier_id=tier_speaker)

    for i in range(len(asr["output_data"]["speaker_turns"])):
        start = int(1000 * asr["output_data"]["speaker_turns"][i]["start"])
        end = int(1000 * asr["output_data"]["speaker_turns"][i]["end"])

        if start >= end:
            continue

        if i in topic_dict:
            eaf.add_annotation(
                tier_speaker, start=start, end=end, value=topic_dict[i]["topic"]
            )
        else:
            eaf.add_annotation(
                tier_speaker, start=start, end=end, value="None"
            )

    return eaf


def create_spk_attr_tier(eaf, input_pickle, asr_pickle):
    logging.info(f"Reading: {input_pickle}")

    spk_attr = pickle.load(open(input_pickle, "rb"))
    asr = pickle.load(open(asr_pickle, "rb"))

    assert len(spk_attr["output_data"]) == len(asr["output_data"]["speaker_turns"])

    # create tier
    tier_speaker = "active_speaker"
    eaf.add_tier(tier_id=tier_speaker)

    tier_speaker = "speaker_role"
    eaf.add_tier(tier_id=tier_speaker)

    tier_speaker = "situation"
    eaf.add_tier(tier_id=tier_speaker)

    for segment in spk_attr["output_data"]:
        start = int(1000 * segment["start"])
        end = int(1000 * segment["end"])

        if start >= end:
            continue

        eaf.add_annotation(
            "active_speaker", start=start, end=end, value="Yes" if segment["active"] else "No"
        )
        eaf.add_annotation(
            "speaker_role", start=start, end=end, value=segment["role_l1"]
        )
        eaf.add_annotation(
            "situation", start=start, end=end, value=segment["situation"]
        )

    return eaf


def create_vlm_tier(eaf, input_pickle, shot_pickle, tier):
    logging.info(f"Reading: {input_pickle}")

    ## Add VLM tier
    tier_speaker = tier
    eaf.add_tier(tier_id=tier_speaker)

    vlm = pickle.load(open(input_pickle, "rb"))
    labels = vlm["output_data"]["labels"]   ## Label vocabulary
    responses = vlm["output_data"]["responses"]  ## (Labels x Time/Frames binary matrix)
    times = vlm["output_data"]["times"] ## Time stamps at 1 fps
    shots = pickle.load(open(shot_pickle, "rb"))

    for shot in shots["output_data"]["shots"]:
        st = shot["start"]
        en = shot["end"]
        start = int(1000 * shot["start"])
        end = int(1000 * shot["end"])

        if start == end:
            continue

        ## Get all time indices within the shot
        shot_inds = [i for i, t in enumerate(times) if t >= st and t <= en]

        ## Check if there are no responses within the shot
        if len(shot_inds) == 0:
            eaf.add_annotation(
                tier, start=start, end=end, value="None"
            )
            continue

        ## Get all labels within the shot and take majority vote
        ## There can be multiple labels for each frame, concatenate names of the labels separated by underscore
        ## Take a majority vote over the shot
        shot_labels = []
        for i in shot_inds:
            shot_labels.append("_".join([labels[j] for j in range(len(labels)) if responses[j][i] == 1]))
        
        ## Take majority vote
        shot_label = max(set(shot_labels), key = shot_labels.count)

        eaf.add_annotation(
            tier, start=start, end=end, value=shot_label
        )

    return eaf


def create_audio_clf_tier(eaf, input_pickle, data_pickle, tier):
    logging.info(f"Reading: {input_pickle}")

    ## Add VLM tier
    tier_clf = tier+"_clf"
    eaf.add_tier(tier_id=tier_clf)
    tier_prob = tier+"_prob"
    eaf.add_tier(tier_id=tier_prob)

    audioclf = pickle.load(open(input_pickle, "rb"))["output_data"]
    if "shot" in tier:
        segments = pickle.load(open(data_pickle, "rb"))["output_data"]["shots"]
    else:
        segments = pickle.load(open(data_pickle, "rb"))["output_data"]["speaker_turns"]

    
    assert len(audioclf) == len(segments)

    for i, segment in enumerate(segments):
        start = int(1000 * segment["start"])
        end = int(1000 * segment["end"])

        if start >= end:
            continue

        eaf.add_annotation(
            tier_clf, start=start, end=end, value=", ".join(audioclf[i]["top3_label"])
        )

        prob_strs = [str(round(p, 2)) for p in audioclf[i]["top3_label_prob"]]

        eaf.add_annotation(
            tier_prob, start=start, end=end, value=", ".join(prob_strs)
        )


    return eaf
    


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

    # create elan object
    eaf = create_elan_file(video_name=args.input)

    videoname = os.path.splitext(os.path.basename(args.input))[0]

    pkl_dir = os.path.join("/nfs/data/fakenarratives/202409_corpus/results_pkl/", args.channel, videoname)

    # add shot tier
    if args.shots:
        if "transnet_shotdetection.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find transnet_shotdetection.pkl for {videoname}")
        else:
            eaf = create_shot_tier(
                eaf, os.path.join(pkl_dir, "transnet_shotdetection.pkl")
            )

    if args.turns:
        if "asr_whisperx.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find asr_whisperx.pkl for {videoname}")
        else:
            eaf = create_turns_tier(eaf, os.path.join(pkl_dir, "asr_whisperx.pkl"))

    if args.topic:
        tpc_dir = os.path.join("analysis_text/topics", args.channel.lower(),"pkls")
        if f"{videoname}_turn_topics.pkl" not in os.listdir(tpc_dir):
            logging.error(f"Cannot find {videoname}_turn_topics.pkl for {videoname}")
        else:
            eaf = create_topic_tier(eaf, 
                                    os.path.join(tpc_dir, f"{videoname}_turn_topics.pkl"), 
                                    os.path.join(pkl_dir, f"asr_whisperx.pkl")
            )
    
    if args.spk_attr:
        if "speaker_turns_meta.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find speaker_turns_meta.pkl for {videoname}")
        else:
            eaf = create_spk_attr_tier(eaf, 
                                       os.path.join(pkl_dir, "speaker_turns_meta.pkl"),
                                       os.path.join(pkl_dir, f"asr_whisperx.pkl")
            )

    if args.vlm:
        if "vlm_social_roles.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find vlm_social_roles.pkl for {videoname}")
        else:
            eaf = create_vlm_tier(eaf, 
                                       os.path.join(pkl_dir, "vlm_social_roles.pkl"),
                                       os.path.join(pkl_dir, f"transnet_shotdetection.pkl"),
                                       "social_roles"
            )
        
        if "vlm_locations.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find vlm_locations.pkl for {videoname}")
        else:
            eaf = create_vlm_tier(eaf, 
                                       os.path.join(pkl_dir, "vlm_locations.pkl"),
                                       os.path.join(pkl_dir, f"transnet_shotdetection.pkl"),
                                       "locations"
            )
        
        if "vlm_events.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find vlm_events.pkl for {videoname}")
        else:
            eaf = create_vlm_tier(eaf, 
                                       os.path.join(pkl_dir, "vlm_events.pkl"),
                                       os.path.join(pkl_dir, f"transnet_shotdetection.pkl"),
                                       "events"
            )

    if args.audioclf:
        if "shot_audioClf.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find shot_audioClf.pkl for {videoname}")
        else:
            eaf = create_audio_clf_tier(eaf, 
                                       os.path.join(pkl_dir, "shot_audioClf.pkl"),
                                       os.path.join(pkl_dir, f"transnet_shotdetection.pkl"),
                                       "shot_audio"
            )

        if "whisperxspeaker_audioClf.pkl" not in os.listdir(pkl_dir):
            logging.error(f"Cannot find whisperxspeaker_audioClf.pkl for {videoname}")
        else:
            eaf = create_audio_clf_tier(eaf, 
                                       os.path.join(pkl_dir, "whisperxspeaker_audioClf.pkl"),
                                       os.path.join(pkl_dir, f"asr_whisperx.pkl"),
                                       "speaker_audio"
            )

    # write elan file
    # print(videoname)
    output_file = os.path.join(args.output, f"{videoname}.eaf")

    os.makedirs(args.output, exist_ok=True)

    logging.info(f"Writing: {output_file}")
    to_eaf(file_path=output_file, eaf_obj=eaf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
