#!/bin/sh

YEAR=$1

NEWSCHANNELS=(Welt HeuteJournal BildTV Tagesschau CompactTV)
# NEWSCHANNELS=(Tagesschau CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    for vid in $videos
    do
        python pickle_to_eaf.py -i $vid -o corpus_outputs/elan_202409/$c -c $c --shots --turns --topic --spk_attr --vlm --audioclf
    done
done