#!/usr/bin/bash
#SBATCH --nodes=1
##SBATCH -w devbox2
##SBATCH -s
#SBATCH -J BLIP_fakenarratives
#SBATCH -G 1
#SBATCH --mem 32G  
#SBATCH -c 2
##SBATCH -o /nfs/home/muellerer/fakenarratives/slurm

FPS=1
NEWSCHANNELS=(Tagesschau)  # (BildTV, CompactTV, HeuteJournal, Tagesschau)

QUERIES=("queries/actions.csv", "queries/news_events.csv", "queries/news_roles.csv")

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls -d /nfs/data/fakenarratives/202306_corpus/videos/$c/*.mp4)
    python blip_object_classification.py --videos $videos --output /nfs/data/fakenarratives/202306_corpus/results_pkl/$c -q queries/*csv --fps $FPS
done
