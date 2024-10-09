#!/bin/sh

YEAR=$1

NEWSCHANNELS=(Tagesschau Welt HeuteJournal CompactTV BildTV)

for c in ${NEWSCHANNELS[*]}
do
# echo %%%% $CHANNEL
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_face/pickle_faceanalysis.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c
done