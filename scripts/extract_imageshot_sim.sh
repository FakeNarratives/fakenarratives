#!/bin/sh

YEAR=$1
CHANNEL=$2
WORKERS=$3

# NEWSCHANNELS=(BildTV Tagesschau Welt HeuteJournal CompactTV)
MODELS=(siglip convnextv2 places)

# for c in ${NEWSCHANNELS[*]}
# do
# echo %%%% $CHANNEL
for m in ${MODELS[*]}
do
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$CHANNEL/*)
    python analysis_visual/pickle_imageShot_similarity.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$CHANNEL -w $WORKERS -m $m 
done