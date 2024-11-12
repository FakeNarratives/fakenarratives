import xml.etree.ElementTree as ET
import json
import os
from collections import Counter


## Change this function to add more tiers to the JSON output
def import_timelines_from_eaf(xmlfile, label_counters):
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    # time spans
    timeslots = {}
    for timeslot in root.findall("TIME_ORDER/TIME_SLOT"):
        timeslots[timeslot.attrib["TIME_SLOT_ID"]] = timeslot.attrib

    # Timeline structure
    timelines = {"video_fn": os.path.basename(root.find("./HEADER/MEDIA_DESCRIPTOR[@MEDIA_URL]").get("MEDIA_URL")),
                 "shots": [], "speaker_turn": [],
                 "story": [], "FEP": [], "Strategy": [], "evaluative-talk": []}

    # Process Tiers
    for timeline in root.findall("TIER"):
        tier_id = timeline.attrib["TIER_ID"]
        
        # Category and label for each annotation based on tier prefix
        if tier_id.startswith("story"):
            label = tier_id.split(":", 1)[-1].lower()
            category = "story"
        elif tier_id.startswith("FEP"):
            label = tier_id.split(":", 1)[-1].lower()
            category = "FEP"
        elif tier_id.startswith("Strategy"):
            label = tier_id.split(":", 1)[-1].lower()
            category = "Strategy"
        elif tier_id.startswith("evaluative-talk"):
            label = tier_id.split(":", 1)[-1].lower()
            category = "evaluative-talk"
        elif tier_id.startswith("shots"):
            category = "shots"
        elif tier_id.startswith("speaker_turn"):
            category = "speaker_turn"
        else:
            continue

        # Annotations for each tier
        timeline_segments = []
        for annotation in timeline.findall("ANNOTATION/ALIGNABLE_ANNOTATION"):
            start_time = timeslots[annotation.attrib["TIME_SLOT_REF1"]]["TIME_VALUE"]
            end_time = timeslots[annotation.attrib["TIME_SLOT_REF2"]]["TIME_VALUE"]

            if category == "shots" or category == "speaker_turn":
                label = annotation.find("ANNOTATION_VALUE").text
            else:
                # Counter for each annotation in the tiers (except shots and speaker_turn)
                label_counters[category][label] += 1

            timeline_segments.append(
                {"start": int(start_time) / 1000, "end": int(end_time) / 1000, "label": label}
            )

        timelines[category].append({"name": tier_id, "segments": timeline_segments})

    return timelines

# Directory paths
annot_dir = "annotations/"
result_dir = "annotations_new/"

# Counters
label_counters = {
    "story": Counter(),
    "FEP": Counter(),
    "Strategy": Counter(),
    "evaluative-talk": Counter()
}

# Loop over all annotations
for xfile in os.listdir(annot_dir):
    if xfile.endswith(".eaf"):
        # Generate JSON output with annotations per file
        timelines = import_timelines_from_eaf(os.path.join(annot_dir, xfile), label_counters)
        json.dump(timelines, open(os.path.join(result_dir, xfile.replace(".eaf", ".json")), "w"), indent=4)

# Statistics of manual annotations
print("Overall Statistics for Unique Labels and Annotation Counts:")
for category, counter in label_counters.items():
    unique_labels_count = len(counter)
    total_annotations = sum(counter.values())
    print(f"Category '{category}':")
    print(f"  Total Unique Labels: {unique_labels_count}")
    print(f"  Total Annotations: {total_annotations}")
    print(f"  Label Counts: {dict(counter)}\n")
