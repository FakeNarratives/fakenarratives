## Setup environments

- conda env create -f environment_face.yml

- conda env create -f environment_emotion.yml

conda run -n faken_v python visual_analysis/pickle_shotdetection.py --videos "$video" --pkl_dir test_videos/features/

conda run -n faken_iface -v python pickle_facedetection.py --videos "$video" --pkl_dir test_videos/features/

## Emotion, gaze and face tracking runs on face detection output

conda run -n deepface -v python pickle_faceemotion.py --videos "$video" --pkl_dir test_videos/features/

conda run -n faken_iface -v python pickle_headgaze.py --videos "$video" --pkl_dir test_videos/features/

conda run -n faken_iface -v python pickle_facetracking.py --videos "$video" --pkl_dir test_videos/features/

conda run -n faken_iface -v python pickle_light-asd.py --videos "$video" --pkl_dir test_videos/features/

conda run -n faken_iface -v python pickle_faceclustering.py --videos "$video" --pkl_dir test_videos/features/

conda run -n faken_iface -v python pickle_faceanalysis.py --videos "$video" --pkl_dir test_videos/features/