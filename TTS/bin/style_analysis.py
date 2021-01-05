#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import time
import traceback

import numpy as np
import torch

import pickle
from random import randrange
from torch.utils.data import DataLoader
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.layers.losses import TacotronLoss
from TTS.tts.utils.distribute import (DistributedSampler,
                                      apply_gradient_allreduce,
                                      init_distributed, reduce_tensor)
from TTS.tts.utils.generic_utils import setup_model, check_config_tts
from TTS.tts.utils.io import save_best_model, save_checkpoint
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import (get_speakers, load_speaker_mapping,
                                    save_speaker_mapping)
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.console_logger import ConsoleLogger
from TTS.utils.generic_utils import (KeepAverage, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     remove_experiment_folder, set_init_dict)
from TTS.utils.io import copy_config_file, load_config
from TTS.utils.radam import RAdam
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import (NoamLR, adam_weight_decay, check_update,
                                gradual_training_scheduler, set_weight_decay,
                                setup_torch_training_env)

from TTS.tts.models.tacotron2 import Tacotron2 
from TTS.tts.utils import *
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.tts.utils.text import text_to_sequence, phoneme_to_sequence
from TTS.tts.utils.text.symbols import symbols, phonemes
import torch



def text_to_seqvec(text, CONFIG):
    text_cleaner = [CONFIG.text_cleaner]
    # text ot phonemes to sequence vector
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(text, text_cleaner, CONFIG.phoneme_language,
                                CONFIG.enable_eos_bos_chars,
                                tp=CONFIG.characters if 'characters' in CONFIG.keys() else None),
            dtype=np.int32)
    else:
        seq = np.asarray(text_to_sequence(text, text_cleaner, tp=CONFIG.characters if 'characters' in CONFIG.keys() else None), dtype=np.int32)
    return seq


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
    return tensor

def id_to_torch(speaker_id, cuda=False):
    if speaker_id is not None:
        speaker_id = np.asarray(speaker_id)
        speaker_id = torch.from_numpy(speaker_id).unsqueeze(0)
    if cuda:
        return speaker_id.cuda().type(torch.long)
    return speaker_id.type(torch.long)

def compute_style_mel(style_wav, ap, cuda=False):
    style_mel = torch.FloatTensor(ap.melspectrogram(
        ap.load_wav(style_wav, sr=ap.sample_rate))).unsqueeze(0)
    if cuda:
        return style_mel.cuda()
    return style_mel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.'
    )
    parser.add_argument('--config',
                        type=bool,
                        default=False,
                        help='Do not verify commit integrity to run training.')
    parser.add_argument('--experiment_folder',
                        type=str,
                        default='',
                        help='Do not verify commit integrity to run training.')

    args = parser.parse_args()

    c = load_config(args.config_path)

    use_cuda = True

    global meta_data_train, meta_data_eval, symbols, phonemes

    # Audio processor
    ap = AudioProcessor(**c.audio)
    if 'characters' in c.keys():
        symbols, phonemes = make_symbols(**c.characters)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    # load data instances
    meta_data_train, meta_data_eval = load_meta_data(c.datasets)

    # set the portion of the data used for training
    if 'train_portion' in c.keys():
        meta_data_train = meta_data_train[:int(len(meta_data_train) * c.train_portion)]
    if 'eval_portion' in c.keys():
        meta_data_eval = meta_data_eval[:int(len(meta_data_eval) * c.eval_portion)]

    num_speakers = 4
    speaker_embedding_dim = None
    speaker_mapping = None

    model = setup_model(num_chars, num_speakers, c, speaker_embedding_dim)

    N = c['gst']['gst_style_tokens']
    train_feats = np.zeros((len(meta_data_train), N))
    valid_feats = np.zeros((len(meta_data_val), N))

    for i in range(len(meta_data_train)):
        style_wave = meta_data_train[i][0]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        _, logits = model.gst_layer(style_mel)
        train_feats[i] = logits.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
        with open(arg.experiment_folder + 'train_feats.pkl', rb) as f:
            pickle.dump(train_feats, f)
        with open(arg.experiment_folder + 'train_meta.pkl', rb) as f:
            pickle.dump(meta_data_train, f)


    for i in range(len(meta_data_val)):
        style_wave = meta_data_val[i][0]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        _, logits = model.gst_layer(style_mel)
        valid_feats[i] = logits.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
        with open(arg.experiment_folder + 'val_feats.pkl', rb) as f:
            pickle.dump(valid_feats, f)
        with open(arg.experiment_folder + 'val_meta.pkl', rb) as f:
            pickle.dump(meta_data_val, f)