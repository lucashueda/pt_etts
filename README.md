# Portuguese Expressive Text-to-Speech

In this project we aim to control expressiveness in text to speech system. (On going)

(On going)

We are still in experiments with multi speaker tacotron2 model. We have implemented 2 different modified Tacotron2 model with speaker simple embedding. The first one inject the embedding toguether with the word embeddings and the second one after the LSA (local sensitive attention). In the experiments the second implementation shows better results with voice synthesis around the iteration 30000 (almost 30 epochs with a subset of 60% VCTK).

# Docker
It is not guarateed that it works, but the file is already there.

# How to run

Actually we just did experiments using VCTK, so the data prep stage is working just for VCTK. In future we aim to make differents data process, and one more generic to be able to be edited.

- 1 - Data preparation
<code> python3 data_preparation.py --input-dir='' --output_dir='' </code>

- 2 - Train
<coode> python3 train.py --output_dir='' --log_dir='' --checkpoint_path='' --model='Tacotron2SE' </code>

- 3 - Synthesis (griffin lim)
There will be a Parallel WaveGan vocoder but with private pre-trained model

<code> python3 synthesis.py --output_dir='' --model='Tacotron2SE' --checkpoint_path='' </code>

# Author

**Lucas Hideki Ueda (lucashueda@gmail.com)**

# Code references

- https://github.com/NVIDIA/tacotron2
- https://github.com/r9y9/tacotron_pytorch
- https://github.com/keithito/tacotron
- https://github.com/espnet/espnet
