#!/bin/sh

YEAR=$1
# CHANNEL=$2
WORKERS=$3

NEWSCHANNELS=(Welt BildTV Tagesschau HeuteJournal CompactTV)
MODELS=(siglip convnextv2 places)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    for m in ${MODELS[*]}
    do
        videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
        python analysis_visual/pickle_imageShot_similarity.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c -w $WORKERS -m $m -r -n 5
    done
done