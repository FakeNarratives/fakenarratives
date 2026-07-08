#!/bin/sh

YEAR=$1
WORKERS=$2
CHANNEL=$3

# NEWSCHANNELS=(BildTV Tagesschau Welt HeuteJournal CompactTV)

# for c in ${NEWSCHANNELS[*]}
# do
echo %%%% $CHANNEL
videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$CHANNEL/*)
python analysis_visual/pickle_shot_scale.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$CHANNEL -w $WORKERS -c square -r
# done