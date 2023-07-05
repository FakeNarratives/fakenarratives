# FakeNarratives

This repository contains code to extract auditory, textual, and visual features from videos used in the BMBF-funded project [*FakeNarratives*](https://fakenarratives.github.io)

**Table of Content**

- [Auditory Feature Extraction](#auditory-feature-extraction)
- [Textual Feature Extraction](#textual-feature-extraction)
- [Visual Feature Extraction](#visual-feature-extraction)
    - [Event Classification](#event-classification)
    - [Geolocation Estimation](#geolocation-estimation)
    - [Optical Character Recognition](#optical-character-recognition)

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
python pickle_VisE_outputs.py --cfg VisE/resources/VisE-D/models/VisE_CO_cos.yml --videos /PATH/TO/VIDEOS --output /PATH/TO/OUTPUT_FOLDER
~~~

For optional parameters, we refer to the file [```pickle_VisE_outputs.py```](pickle_VisE_outputs.py)


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

