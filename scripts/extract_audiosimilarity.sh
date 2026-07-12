#!/bin/sh

YEAR=$1
MODEL=$2

NEWSCHANNELS=(Welt BildTV Tagesschau HeuteJournal CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_audio/pickle_audio_similarity.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c -m $MODEL -n 5
done