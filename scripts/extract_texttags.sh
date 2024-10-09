#!/bin/sh

YEAR=$1

NEWSCHANNELS=(HeuteJournal)

for c in ${NEWSCHANNELS[*]}
do
# echo %%%% $CHANNEL
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_text/pickle_texttagger.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c
done