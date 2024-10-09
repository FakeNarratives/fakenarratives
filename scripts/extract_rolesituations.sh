#!/bin/sh

YEAR=$1

NEWSCHANNELS=(BildTV Tagesschau Welt HeuteJournal CompactTV)

# for c in ${NEWSCHANNELS[*]}
# do
#     echo %%%% $c
#     videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
#     python analysis_multimodal/srr_nsr_code/pickle_rolesituations.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c --rf --nsr
# done

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/${YEAR}_corpus/videos/$c/*)
    python analysis_multimodal/srr_nsr_code/pickle_rolesituations.py -v $videos -o /nfs/data/fakenarratives/${YEAR}_corpus/results_pkl/$c --gb --srr -m analysis_multimodal/srr_nsr_code/models
done