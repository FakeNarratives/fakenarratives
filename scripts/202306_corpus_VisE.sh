#!/usr/bin/bash
#SBATCH --nodes=1
##SBATCH -w devbox2
##SBATCH -s
#SBATCH -J VisE_fakenarratives
#SBATCH -G 1
#SBATCH --mem 16G  
##SBATCH -c 24
##SBATCH -o /nfs/home/muellerer/fakenarratives/slurm

FPS=5
NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/202306_corpus/videos/$c/*.mp4)
    python pickle_VisE_outputs.py --cfg VisE/resources/VisE-D/models/VisE_CO_cos.yml --videos $videos --output /nfs/data/fakenarratives/202306_corpus/results_pkl/$c --fps $FPS
done
