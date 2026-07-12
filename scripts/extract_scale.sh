#!/bin/sh

YEAR=$1
WORKERS=$2
# CHANNEL=$3
CROP=$3

NEWSCHANNELS=(BildTV Tagesschau Welt HeuteJournal CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_visual/pickle_shot_scale.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c -w $WORKERS -c $CROP
done