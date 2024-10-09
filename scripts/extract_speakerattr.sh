#!/bin/sh

YEAR=$1

NEWSCHANNELS=(Welt BildTV Tagesschau HeuteJournal CompactTV)
# NEWSCHANNELS=(CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_audio/pickle_speakerattr_outputs.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c
done