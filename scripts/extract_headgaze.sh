#!/bin/sh

YEAR=$1
CHANNEL=$2
WORKERS=$3
BATCH=$4

# NEWSCHANNELS=(BildTV Tagesschau Welt HeuteJournal CompactTV)

# for c in ${NEWSCHANNELS[*]}
# do
echo %%%% $CHANNEL
videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$CHANNEL/*)
python analysis_face/pickle_headgaze.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$CHANNEL -w $WORKERS -b $BATCH
# done