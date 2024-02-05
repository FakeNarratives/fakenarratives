#!/usr/bin/bash

NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    inputs=$(ls -d /nfs/data/fakenarratives/202306_corpus/results_pkl/$c/*)

    for i in $inputs
    do 
        echo "python export_graphml.py --input $i --output /nfs/data/fakenarratives/202306_corpus/results_graphml/$c --pyviz"
        python export_graphml.py --input $i --output /nfs/data/fakenarratives/202306_corpus/results_graphml/$c --pyviz
        echo %%
    done
done