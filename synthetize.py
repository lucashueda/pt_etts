'''
    This script performs audio synthesis given a text, a model + model checkpoint and the synthetizer

'''

import os
import time
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default = 'Tacotron2',
                        help='The model to be used')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-v', '--vocoder', type=str, default='Griffin-Lim',
                        help='A valid vocoder between: "Griffin-Lim" and "WaveGlow"')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    