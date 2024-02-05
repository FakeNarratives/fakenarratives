NEWSCHANNELS=(BildTV CompactTV HeuteJournal Tagesschau)

for c in ${NEWSCHANNELS[*]}
do
    echo %%%% $c
    videos=$(ls /nfs/data/fakenarratives/202306_corpus/results_pkl/$c/)

    for v in $videos
    do
        in_folder=/nfs/data/fakenarratives/202306_corpus/results_pkl/$c/$v
        out_folder=/nfs/data/fakenarratives/202306_corpus/results_eaf_speakers/$c/$v

        # echo $in_folder
        # echo $out_folder
        # echo %%%%

        python pickle_to_eaf.py --input $in_folder --output $out_folder --shots --speaker
    done
done
