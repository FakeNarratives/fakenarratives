#!/usr/bin/bash
#SBATCH --nodes=1
##SBATCH -w devbox2
##SBATCH -s
#SBATCH -J VisE_fakenarratives
#SBATCH -G 0
##SBATCH -c 24
#SBATCH -o /nfs/home/muellerer/fakenarratives/slurm

FILES=["/nfs/data/fakenarratives/BildTV/videos/20220118_Corona_Demos_Randale_und_Aggression_ydzy-emLHfY.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220118_Corona_Zahlen_in_Deutschland_Diese_pDeXHOhVXz0.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220118_Corona_Demos_Randale_und_Aggression_ydzy-emLHfY.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220120_Omikron_Welle_Diese_Impfpflicht_ist_pWZDF3rJ744.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220114_Ungeimpfter_darf_Raum_nicht_betreten_BZDHKarAQcA.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220104_Schluss_mit_dem_Corona_Lockdown_1odqvo1zKzI.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220113_New_York_pfeift_auf_die_Mfh22Av5-jI.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220105_Corona_Regeln_Unsere_Freiheit_gerät_0RM8KQi3Muk.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220114_Omikron_In_vier_Wochen_hatte_PIA-koNrP1M.mp4", \
    "/nfs/data/fakenarratives/BildTV/videos/20220120_Omikron_Impfpflicht_hat_keinen_Sinn_WAwcNYwvcmE.mp4"]

for f in $FILES
do
    python pickle_VisE_outputs.py --cfg VisE/resources/VisE-D/models/VisE_CO_cos.yml --videos $f --output /nfs/data/fakenarratives/BildTV/results/20230202_VisE
done


FILES=["/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_10_kX5DF1JDnqg.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_13_hLfcV8qf4fY.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_12_jRZ-7yi3gCw.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_11__kAfWRM5wZQ.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_17_HU1GFKcRV70.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_27_WU7Ga9cXvc4.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_19__PFsywDqSYs.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_20_HEdDHMVfJxM.mp4", \
    "/nfs/data/fakenarratives/CompactTV/videos/compacttv_2022_01_18_zwJZgvrfy4c.mp4"]

for f in $FILES
do
    python pickle_VisE_outputs.py --cfg VisE/resources/VisE-D/models/VisE_CO_cos.yml --videos $f --output /nfs/data/fakenarratives/CompactTV/results/20230202_VisE
done


FILES=["/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220107-2024-3000.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220110-2020-5400.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220108-2023-3400.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220104-2020-5700.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220103-2020-2800.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220106-2021-1700.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220102-2021-1800.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220120-2022-0200.webl.h264.mp4", \
    "/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220109-2025-2800.webl.h264.mp4"]

for f in $FILES
do
    python pickle_VisE_outputs.py --cfg VisE/resources/VisE-D/models/VisE_CO_cos.yml --videos $f --output /nfs/data/fakenarratives/tagesschau/results/20230202_VisE
done