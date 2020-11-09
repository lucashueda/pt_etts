# Portuguese Expressive Text-to-Speech

In this project we aim to control expressiveness in text to speech system. (On going)

(On going)

We are still in experiments with multi speaker tacotron2 model. We have implemented 2 different modified Tacotron2 model with speaker simple embedding. The first one inject the embedding toguether with the word embeddings and the second one after the LSA (local sensitive attention). In the experiments the second implementation shows better results with voice synthesis around the iteration 30000 (almost 30 epochs with a subset of 60% VCTK).


This project in mainly based on NVIDIA implementation of tacotron2 ([https://github.com/NVIDIA/tacotron2]([1])) and we tried to keep some basic structure of ESPNET framework ([https://github.com/espnet/espnet]([4])). 

# Docker
In progress to develop a docker image to better reproduce this project. At this time you have the requeriments.txt with all packages and versions used, also yo uneed PyTorch that is not specified.

# How to run

To run all the code you need to specify the dataset, preprocess data and then train the model. The parameters of all pipelines are defined by the .yaml file in "conf/" directory
that you specify which one to use in arguments of each step.

## Configuration

All paramaters of feature extraction, model dimensions, model training and checkpoint settings need to be in a .yaml file. A default example is provided in "conf/" directory, but you can specify your own parameters.

## Dataset 

Actually we just did experiments using VCTK, so the data prep stage is working just for VCTK. In future we aim to make differents data process, and one more generic to be able to be edited. We will try to make the data preprocess sufficiently generic to be able to be performed in any dataset.

## Script

To run the pipeline you need to run these 3 scripts below:

- 1 - Data preparation
<code> python3 data_preparation.py --input-dir='' --output_dir='' </code>

- 2 - Train
<code> python3 train.py --output_dir='' --log_dir='' --checkpoint_path='' --model='Tacotron2SE' </code>

- 3 - Synthesis (griffin lim)
There will be a Parallel WaveGan vocoder but with private pre-trained model
<code> python3 synthesis.py --output_dir='' --model='Tacotron2SE' --checkpoint_path='' </code>

# Author

**Lucas Hideki Ueda (lucashueda@gmail.com)**

# Code references

- [1] https://github.com/NVIDIA/tacotron2
- [2] https://github.com/r9y9/tacotron_pytorch
- [3] https://github.com/keithito/tacotron
- [4] https://github.com/espnet/espnet
