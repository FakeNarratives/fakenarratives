#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH -J fakenarratives
#SBATCH -G 1

NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/202306_corpus/videos/$c/*.mp4)
    python pickle_headpose.py --videos $videos --output /nfs/data/fakenarratives/202306_corpus/results_pkl/$c
done