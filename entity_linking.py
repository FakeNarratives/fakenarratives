from wikidata import get_entity_response, get_wikidata_entries


def fix_entity_types(all_linked_entities, event_list):
    entity_info = {}

    for linked_entities in all_linked_entities:
        for i in range(len(linked_entities)):
            wd_id = linked_entities[i]['wd_id']
            if wd_id not in entity_info:
                entity_info[wd_id] = get_entity_response(wikidata_id=wd_id)

            if wd_id in event_list:
                is_event = True
            else:
                is_event = False

            is_person = False
            is_location = False

            information = ["wikipedia_url", "entityDescription", "wdimage"]
            for b in entity_info[wd_id]["bindings"]:
                if "instance" in b and "value" in b["instance"] and b["instance"]["value"].endswith("/Q5"):
                    is_person = True

                if "coordinate" in b and "value" in b["coordinate"]:
                    is_location = True

                for info_tag in information:
                    if info_tag in b and "value" in b[info_tag]:
                        linked_entities[i][info_tag] = b[info_tag]["value"]
                    else:
                        linked_entities[i][info_tag] = ""


            if is_location:
                linked_entities[i]["type"] = "LOC"
            if is_person:  # NOTE higher priority if an entity is an instance of person then it cannot be a location
                linked_entities[i]["type"] = "PER"
            if is_event:  # NOTE highest priority as the entity is covered by EventKG
                linked_entities[i]["type"] = "EVENT"
            if not (is_location or is_person or is_event):
                linked_entities[i]["type"] = "MISC"

    return all_linked_entities


def link_annotations(spacy_annotations, wikifier_annotations):
    POSSIBLE_SPACY_TYPES = ['PER', 'ORG','LOC', 'MISC']
    linked_entities = []
    for spacy_annots, wikifier_anno in zip(spacy_annotations, wikifier_annotations):
        sent_annots = []
        for spacy_anno in spacy_annots:
            # skip all entities with 0 or 1 characters or not in selected spacy types
            if len(spacy_anno['text']) < 2 or spacy_anno['type'] not in POSSIBLE_SPACY_TYPES:
                continue
                
            related_wikifier_entries = get_related_wikifier_entry(spacy_anno, wikifier_anno)

            # if no valid wikifier entities were found, try to find entity based on string using <wbsearchentities>
            if len(related_wikifier_entries) < 1:
                # get wikidata id for extrated text string from spaCy NER
                entity_candidates = get_wikidata_entries(entity_string=spacy_anno['text'], limit_entities=1, language="de")

                # if also no match continue with next entity
                if len(entity_candidates) < 1:
                    entity_candidate = {
                    **{
                        'wd_id': None,
                        'wd_label': None,
                        'disambiguation': None,
                        'link': None
                    },
                    **spacy_anno,
                    }
                    sent_annots.append(entity_candidate)
                    continue

                # take the first entry in wbsearchentities (most likely one)
                entity_candidate = {
                    **{
                        'wd_id': entity_candidates[0]['id'],
                        'wd_label': entity_candidates[0]['label'],
                        'disambiguation': 'wbsearchentities',
                        'url' : entity_candidates[0]['url'],
                    },
                    **spacy_anno,
                }
                sent_annots.append(entity_candidate)
            else:
                highest_PR = -1
                best_wikifier_candidate = related_wikifier_entries[0]
                for related_wikifier_entry in related_wikifier_entries:
                    # print(related_wikifier_entry['title'], related_wikifier_entry['pageRank_occurence'])
                    if related_wikifier_entry['pageRank_occurence'] > highest_PR:
                        best_wikifier_candidate = related_wikifier_entry
                        highest_PR = related_wikifier_entry['pageRank_occurence']

                entity_candidate = {
                    **{
                        'wd_id': best_wikifier_candidate['wikiDataItemId'],
                        'wd_label': best_wikifier_candidate['secTitle'],
                        'disambiguation': 'wikifier',
                        'url': best_wikifier_candidate['url']
                    },
                    **spacy_anno,
                }
                sent_annots.append(entity_candidate)
        
        linked_entities.append(sent_annots)

    return linked_entities



def get_related_wikifier_entry(spacy_anno, wikifier_annotations, char_tolerance=2, threshold=1e-4):
    # loop through entities found by wikifier
    aligned_candidates = []
    for wikifier_entity in wikifier_annotations:
        if 'secTitle' not in wikifier_entity.keys() or 'wikiDataItemId' not in wikifier_entity.keys():
            continue

        wikifier_entity_occurences = wikifier_entity['support']

        # loop through all occurences of a given entity recognized by wikifier
        for wikifier_entity_occurence in wikifier_entity_occurences:

            if wikifier_entity_occurence['chFrom'] < spacy_anno['start'] - char_tolerance:
                continue

            if wikifier_entity_occurence['chTo'] > spacy_anno['end'] + char_tolerance:
                continue

            # apply very low threshold to get rid of annotation with very low confidence
            if wikifier_entity_occurence['pageRank'] < threshold:
                continue

            aligned_candidates.append({
                **wikifier_entity,
                **{
                    'pageRank_occurence': wikifier_entity_occurence['pageRank']
                }
            })

    return aligned_candidates
