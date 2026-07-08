#!/bin/sh

YEAR=$1
CHANNEL=$2
# NEWSCHANNELS=(BildTV Welt HeuteJournal CompactTV Tagesschau)
# NEWSCHANNELS=(CompactTV)

# for c in ${NEWSCHANNELS[*]}
# do
echo %%%% $CHANNEL
videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$CHANNEL/*)
python analysis_audio/pickle_speakerturns_meta.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$CHANNEL -r
# done