# FakeNarratives

This repository contains code to extract auditory, textual, and visual features from videos used in the BMBF-funded project [*FakeNarratives*](https://fakenarratives.github.io)

**Table of Content**

- [FakeNarratives](#fakenarratives)
  - [Auditory Feature Extraction](#auditory-feature-extraction)
  - [Textual Feature Extraction](#textual-feature-extraction)
  - [Visual Feature Extraction](#visual-feature-extraction)
    - [Event Classification](#event-classification)
    - [Face Clustering](#face-clustering)
    - [Geolocation Estimation](#geolocation-estimation)
    - [InstructBLIP](#intructblip)
    - [Headpose Estimation](#headpose-estimation)
    - [Optical Character Recognition](#optical-character-recognition)
  - [Graph Creation](#graph-creation)

## Auditory Feature Extraction

## Textual Feature Extraction

## Visual Feature Extraction 

### Event Classification

To install all dependencies, please use the following commands:

~~~sh
cd VisE
conda create --name fakenarratives_vise_py38 python=3.8
pip install -r requirements.txt
~~~

To extract features for a given video, please run:

~~~sh
conda activate fakenarratives_vise_py38
cd VisE
python download_resources.py
cd ..
python pickle_VisE_outputs.py --cfg VisE/resources/VisE-D/models/VisE_CO_cos.yml --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER
~~~

For optional parameters, we refer to the file [```pickle_VisE_outputs.py```](pickle_VisE_outputs.py)

### Face Clustering

> **_NOTE:_**  This plugin requires the ```face_analysis.pkl``` that contains faces detected by *insightface*.
> For this purpose, the corresponding *TIB-AV-A* pipeline has been used. 

To install all dependencies, please use the following commands:

~~~sh
pip install numpy
pip install scikit-learn
~~~

For face clustering in a given video, the face_analysis pipeline from TIB-AV-A needs to be executed first. 
Based on the ```face_analysis.pkl``` written by the corresponding 
[*TIB-AV-A*](https://github.com/TIBHannover/tibava-analyser) pipeline, please run:

~~~sh
python pickle_faceclustering.py --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER
~~~

### Geolocation Estimation

To install all dependencies, please use the following commands:

~~~sh
cd semantic_geo_partitioning
conda create --name fakenarratives_semantic_geo_partitioning_py38 python=3.8
pip install -r requirements.txt
~~~

To extract features for a given video, please run:

~~~sh
conda activate fakenarratives_semantic_geo_partitioning_py38
PYTHONPATH=./semantic_geo_partitioning/geo_classification python pickle_geoestimation.py --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER
~~~

### Headpose Estimation

> **_NOTE:_**  This plugin requires the ```face_analysis.pkl``` that contains faces detected by *insightface*.
> For this purpose, the corresponding *TIB-AV-A* pipeline has been used. 

To install all dependencies, please use the following commands:

~~~sh
cd 6DREpNet
conda create --name fakenarratives_headpose_py38 python=3.8
pip install -r requirements.txt
mkdir model
cd model
wget https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth
~~~

To extract features for a given video, the face_analysis pipeline from TIB-AV-A needs to be executed first. 
Based on the ```face_analysis.pkl``` written by the corresponding 
[*TIB-AV-A*](https://github.com/TIBHannover/tibava-analyser) pipeline, please run:

~~~sh
conda activate fakenarratives_headpose_py38
python pickle_headpose.py --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER
~~~

### IntructBLIP

To install all dependencies, please use the following commands:

~~~sh
cd instructblip
conda create --name fakenarratives_instructblip_py311 python=3.11
pip install -r requirements.txt
~~~

To extract features for a given video, please run:

~~~sh
conda activate fakenarratives_instructblip_py311
python blip_object_classification.py --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER --queries /PATH/TO/QUERY/CSVs
~~~

### Optical Character Recognition

To install all dependencies, please use the following commands:

~~~sh
cd mmocr
conda create --name fakenarratives_mmocr_py38 python=3.8
pip install -r requirements.txt
~~~

To extract features for a given video, please run:

~~~sh
cd mmocr
conda activate fakenarratives_mmocr_py38
python ../pickle_mmocr_outputs.py --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER --fps $FPS
~~~

For optional parameters, we refer to the file [```pickle_mmocr_outputs.py```](pickle_mmocr_outputs.py)

## Graph Creation

To install all dependencies, please use the following commands:

~~~sh
cd graph_visualization
conda create --name fakenarratives_graph_py311 python=3.11
pip install -r requirements.txt
~~~

To create a graphlml file based on the extracted features (from .pkl files), please run:

~~~sh
cd graph_visualization
conda activate fakenarratives_graph_py311
python export_graphml.py --input /PATH/TO/RESULT_PKLS --output /PATH/TO/OUTPUT_FOLDER
~~~