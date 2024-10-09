#!/bin/sh

YEAR=$1

NEWSCHANNELS=(BildTV Tagesschau Welt HeuteJournal CompactTV)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_audio/pickle_asr_outputs.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c -b 16
done