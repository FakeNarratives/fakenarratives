import pickle
import os

results_loc = "/nfs/data/fakenarratives/202306_corpus/results_pkl"

elite_persons_sent = {}
organizations = {}
locations = {}
events = {}

for channel in os.listdir(results_loc):
    for vidname in os.listdir(os.path.join(results_loc, channel)):

        if not os.path.exists(os.path.join(results_loc, channel, vidname, "whisperx_ner.pkl")):
            continue

        feature_dict = pickle.load(open(os.path.join(results_loc, channel, vidname, "whisperx_ner.pkl"), 'rb'))

        feature_dict = feature_dict["output_data"]

        for segment in feature_dict["speakerturn_wise"]:
            if segment["tags"] is not None:
                for ent in segment["tags"]:
                    if ent['type'] == "EPER":
                        if ent['wd_label'] not in elite_persons_sent:
                            elite_persons_sent[ent['wd_label']] = ent['wd_id']
                    elif ent['type'] == "ORG":
                        if ent['wd_label'] not in organizations:
                            organizations[ent['wd_label']] = ent['wd_id']
                    elif ent['type'] == "LOC":
                        if ent['wd_label'] not in locations:
                            locations[ent['wd_label']] = ent['wd_id']
                    elif ent['type'] == "EVENT":
                        if ent['wd_label'] not in events:
                            events[ent['wd_label']] = ent['wd_id']



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
