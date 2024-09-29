# Startup guide

## Setting up the dataset files

1. Download the repository
2. Download the following files from the LIRIS-ACCEDE website (https://liris-accede.ec-lyon.fr/):
  - From the Discrete LIRIS-ACCEDE collection:
    - `LIRIS-ACCEDE-data.zip`
    - `LIRIS-ACCEDE-annotations.zip`
  - From the MediaEval 2015 Affective Impact of Movies collection
    - `MEDIAEVAL-data.zip`
    - `MEDIAEVAL-annotations.zip`
    - `MEDIAEVAL-mapping.zip`
3. Unzip all of the zip files

### Moving the videos
   - Move all of the videos from `LIRIS-ACCEDE-data/data/` into `TrAVCA/data/liris_accede/data/`
   - Move all of the videos from `MEDIAEVAL-data/data/` into `TrAVCA/data/liris_accede/data/`

```
TrAVCA
├── data
│   └── liris_accede
│       └── data
│           ├── ACCEDE00000.mp4
│           ├── ACCEDE00001.mp4
│           ├── ...
│           ├── ACCEDE09799.mp4
│           ├── MEDIAEVAL00000.mp4
│           ├── MEDIAEVAL00001.mp4
│           └── ...
└── README.md
```
After this step, there should be 10,900 video files in the `TrAVCA/data/liris_accede/data/` folder.

   
### Moving the text files with the annotations
   - Move the file `ACCEDEsets.txt` from `LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/` into `TrAVCA/data/liris_accede/annotations/`
   - Move the 3 files `MEDIAEVALsets.txt`, `ACCEDEaffect.txt`, and `MEDIAEVALaffect.txt` from `MEDIAEVAL-annotations/annotations/` into `TrAVCA/data/liris_accede/annotations/`
   - Move the 2 files `shots-devset-nl.txt` and `shots-testset-nl.txt` from `MEDIAEVAL-mapping/MEDIAEVAL-mapping/` into `TrAVCA/data/liris_accede/annotations/`

```
TrAVCA
├── data
│   └── liris_accede
│       └── annotations
│           ├── ACCEDEaffect.txt
│           ├── ACCEDEsets.txt
│           ├── MEDIAEVALaffect.txt
│           ├── MEDIAEVALsets.txt
│           ├── shots-devset-nl.txt
│           └── shots-testset-nl.txt
└── README.md
```
After this step, these 6 .txt files should be in the `TrAVCA/data/liris_accede/data/` folder.

## Setting up AudioCLIP
We use AudioCLIP (https://github.com/AndreyGuzhov/AudioCLIP/) as a pretrained model. The AudioCLIP weights are not included in this repository, and need to be downloaded manually.

1. From the AudioCLIP assets folder (https://github.com/AndreyGuzhov/AudioCLIP/tree/master/assets), download the following files:
    - `AudioCLIP-Full-Training.pt`
    - `bpe_simple_vocab_16e6.txt.gz`
2. Move both of these files into the `TrAVCA/pretrained_model/audio_clip/assets/` folder.

```
TrAVCA
├── pretrained_model
│   └── audio_clip
│       └── assets
│           ├── .gitkeep
│           ├── AudioCLIP-Full-Training.pt
│           └── bpe_simple_vocab_16e6.txt.gz
└── README.md
```

## Setting up the conda environment
In order to run the code locally, use the following command to setup the conda environment:

```
conda env create --file=environment.yml
```

## Alternatively: Using Docker
If you want to run the code in a Docker container, you can build the Docker image with the following command:

```
docker build -t travca .
```

To create a container from this image, use the following command:

```
docker run --gpus all --shm-size=4gb -p 8888:8888 -v ${pwd}:/app -it travca
```

# Running the code
## Pre-processing
Before being able to perform training or testing, the data needs to be preprocessed. 

**Warning: Be aware that the preprocessed data will take up about 110 GB of disk space!**

1. Open the preprocessing notebook at `TrAVCA/data/liris_accede/preprocess.ipynb`
2. Ensure that the **TrAVCA** environment is selected as the active kernel
3. Execute all cells of the notebook
   
Note: Depending on your systems' performance, preprocessing may take up to several hours.

## Training and testing

If it is not already active, activate the conda environment:

```
conda activate TrAVCA
```

### Training a model
You can then run the main application using python:
```
python main.py
```

For a list of parameters that can be set, run the help command:
```
python main.py -h
```

### Testing a model
Instead of training a new model, you may also test an existing model. Testing currently only works on the test portions of the implemented datasets. For example, if a model was trained on the AIMT15 dataset, test.py will test the model on the test portion of that dataset. You should specify the path to the checkpoint file  as a parameter. See this example:

```
python test.py --checkpoint_file model/checkpoints/my_model.pth
```

For a list of parameters that can be set, run the help command:
```
python test.py -h
```
