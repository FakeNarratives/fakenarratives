#!/bin/sh

YEAR=$1

NEWSCHANNELS=(BildTV Welt HeuteJournal CompactTV Tagesschau)
# NEWSCHANNELS=(CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_audio/pickle_speakerturns_meta.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c -r
done