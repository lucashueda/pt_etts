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


sys.path.insert(1, '/l/disk1/awstebas/lhueda/github/repo_final/repo_final_final/repo_final_final_final/pt_etts/')

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

    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    # load data instances
    meta_data_train, meta_data_eval = load_meta_data(c.datasets)

    num_speakers = 4
    speaker_embedding_dim = None
    speaker_mapping = None
    num_styles = 2
    model = setup_model(num_chars, num_speakers,num_styles, c, speaker_embedding_dim)
    
    MODEL_PATH = "/l/disk0/lhueda/github/repo_final/repo_final_final/repo_final_final_final/pt_etts/experiments/gst_4speakers_6tokens2heads_style_target_alfredo/best_model.pth.tar"
    cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model'])
    model.eval()

    # N = c['gst']['gst_style_tokens']
    # h = c['gst']['gst_num_heads']
    # N = N*h

    N = c['gst']['gst_embedding_dim']

    train_feats = np.zeros((len(meta_data_train), N))
    valid_feats = np.zeros((len(meta_data_eval), N))

    for i in range(len(meta_data_train)):
        style_wav = meta_data_train[i][1]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        outputs , logits = model.cuda().gst_layer(style_mel)
        # logits = torch.cat(
        #     torch.split(logits, 1, dim=0),
        #     dim=3).squeeze(0)

        #print(style_wav, style_mel.shape, logits, logits.shape)
        train_feats[i] = outputs.squeeze(0).squeeze(0).detach().cpu().numpy()
    with open(args.experiment_folder + 'train_feats.pkl', 'wb') as f:
        pickle.dump(train_feats, f)
    with open(args.experiment_folder + 'train_meta.pkl', 'wb') as f:
        pickle.dump(meta_data_train, f)


    for i in range(len(meta_data_eval)):
        style_wav = meta_data_eval[i][1]
        style_mel = compute_style_mel(style_wav, ap, cuda=True)
        style_mel = numpy_to_torch(style_mel, torch.float, cuda=use_cuda)
        outputs , logits = model.cuda().gst_layer(style_mel)

        # logits  = torch.cat(
        #     torch.split(logits, 1, dim=0),
        #     dim=3).squeeze(0)

        valid_feats[i] = outputs.squeeze(0).squeeze(0).detach().cpu().numpy()
    with open(args.experiment_folder + 'val_feats.pkl', 'wb') as f:
        pickle.dump(valid_feats, f)
    with open(args.experiment_folder + 'val_meta.pkl', 'wb') as f:
        pickle.dump(meta_data_eval, f)
