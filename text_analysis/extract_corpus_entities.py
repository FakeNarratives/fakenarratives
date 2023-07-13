import pickle
import os

results_loc = "/nfs/data/fakenarratives/202306_corpus/results_pkl"
file_path = "../video_list.txt"

elite_persons_sent = {}
organizations = {}
locations = {}
events = {}

with open(file_path, 'r') as f:
    pickle_path = f.read().splitlines()

    for video_name in pickle_path:

        feature_dict = pickle.load(open(os.path.join(results_loc, video_name, "asr_textnlp.pkl"), 'rb'))

        feature_dict = feature_dict["output_data"]

        for segment in feature_dict["ner"]["sentence_wise"]:
            for ent in segment["tags"]:
                if ent[2] == "EPER":
                    if ent[3] not in elite_persons_sent:
                        elite_persons_sent[ent[3]] = ent[1]
                elif ent[2] == "ORG":
                    if ent[3] not in organizations:
                        organizations[ent[3]] = ent[1]
                elif ent[2] == "LOC":
                    if ent[3] not in locations:
                        locations[ent[3]] = ent[1]
                elif ent[2] == "EVENT":
                    if ent[3] not in events:
                        events[ent[3]] = ent[1]



with open("corpus_outputs/202306_corpus/elite_persons.txt", "w") as fw:
    for person, qid in elite_persons_sent.items():
        fw.write(person+"\t"+ qid + "\n")

with open("corpus_outputs/202306_corpus/organizations.txt", "w") as fw:
    for org, qid in organizations.items():
        fw.write(org +"\t"+ qid +"\n")

with open("corpus_outputs/202306_corpus/locations.txt", "w") as fw:
    for loc, qid in locations.items():
        fw.write(loc+"\t"+ qid + "\n")

with open("corpus_outputs/202306_corpus/events.txt", "w") as fw:
    for event, qid in events.items():
        fw.write(event +"\t"+ qid + "\n")
