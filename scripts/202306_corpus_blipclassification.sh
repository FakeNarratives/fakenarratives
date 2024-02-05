#!/usr/bin/bash
#SBATCH --nodes=1
##SBATCH -w devbox2
##SBATCH -s
#SBATCH -J FN_BLIP
#SBATCH -G 1
#SBATCH --mem 32G  
#SBATCH -c 2
##SBATCH -o /nfs/home/muellerer/fakenarratives/slurm

FPS=1
# (BildTV CompactTV HeuteJournal Tagesschau)
NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)  

QUERIES=("queries/actions.csv" "queries/daytime.csv" "queries/environment.csv" "queries/news_events.csv" "queries/news_roles.csv" "queries/weather.csv")

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    echo ${QUERIES[*]}
    videos=$(ls -d /nfs/data/fakenarratives/202306_corpus/videos/$c/*.mp4)
    python blip_object_classification.py --videos $videos --output /nfs/data/fakenarratives/202306_corpus/results_pkl/$c -q ${QUERIES[*]} --fps $FPS
done
