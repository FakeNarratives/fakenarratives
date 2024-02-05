#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH -J fakenarratives

NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls /nfs/data/fakenarratives/202306_corpus/results_pkl/$c/)

    for v in ${videos[*]}
    do
        python pickle_faceanalysis.py --input /nfs/data/fakenarratives/202306_corpus/results_pkl/$c/$v
    done
done