#!/bin/sh

YEAR=$1

NEWSCHANNELS=(Welt HeuteJournal BildTV Tagesschau CompactTV)
# NEWSCHANNELS=(CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_audio/pickle_audioclf_outputs.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c -r
done