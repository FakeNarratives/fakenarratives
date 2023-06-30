NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls /nfs/data/fakenarratives/202306_corpus/videos/$c/)

    for v in $videos
    do        
        mkdir -p /nfs/data/fakenarratives/202306_corpus/results_pkl/$c/${v%.*}
        mkdir -p /nfs/data/fakenarratives/202306_corpus/results_yml/$c/${v%.*}
        # mkdir -p /nfs/data/fakenarratives/202306_corpus/results_eaf/$c/${v%.*}
    done
done


chmod o+w -r /nfs/data/fakenarratives/202306_corpus/