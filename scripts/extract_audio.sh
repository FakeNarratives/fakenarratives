#!/bin/sh

NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau Welt)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/202409_corpus/videos/$c/*)
    python analysis_audio/extract_audios.py -v $videos -o /nfs/data/fakenarratives/202409_corpus/results_pkl/$c
done