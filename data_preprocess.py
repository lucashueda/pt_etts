import argparse
import logging
import os

import librosa
import numpy as np
import yaml
import pandas as pd
import pickle
from src.audio.audio_processing import mel_normalize

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.
    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).
    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

def prepare_directory(out_dir):
  '''
    Função que cria um diretório de saída de arquivos
  '''
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    os.chmod(out_dir, 0o775) # Não tenho certeza se funciona no windows


def main(outdir, config):

    training_files = config['data_training_files']
    validation_files = config['data_validation_files']

    # Creating directory of dataset
    prepare_directory(outdir)
    # Creating subfolders to train and val dataset
    prepare_directory(os.path.join(outdir,'train'))
    prepare_directory(os.path.join(outdir, 'val'))

    
    train_df = pd.read_csv(training_files, sep=',', encoding='latin-1')
    val_df = pd.read_csv(validation_files, sep=',', encoding='latin-1')

    ################################ TRAINING PROCESS ############################################

    actual_store_folder = os.path.join(outdir,'train/0')
    prepare_directory(actual_store_folder)

    # List that will store mel and audio length paths
    mel_dirs = []
    durations = []

    # Normalization statistics, calculated just in training data and used to normalize all data
    counts = {}
    sum_feats = {}
    square_sum_feats = {}

    # Starting trainning files process
    print(f"Saving train features in {os.path.join(outdir, 'train')} folder.")
    for i in range(train_df.shape[0]):

        try:
            print('enter try')
            audio , sr = librosa.load(train_df.wav_path.values[i], sr = config['sampling_rate'])
            print('done reading audio')
            # Trimming silence
            audio = librosa.effects.trim(audio, top_db=config['trim_treshold_in_db'], 
                                        frame_length=config['trim_frame_size'],
                                        hop_length=config['trim_hop_size'])[0]
            audio = np.append([0.]*5*config['trim_hop_size'], audio)
            audio = np.append(audio, [0.]*5*config['trim_hop_size'])

            print('done ')

            durations.append(len(audio)/sr)

            log10mel = logmelfilterbank(audio, config['sampling_rate'], config['filter_length'], 
                                    config['hop_length'], config['win_length'], config['window'],
                                    config['n_mel_channels'], config['mel_fmin'], config['mel_fmax'])
            
            print('done logmel')

            if(config['mel_clip_normalize'] == True):
                log10mel = mel_normalize(log10mel, ref_level_db = config['ref_level_db'], 
                                        max_abs_value = config['max_abs_value'], min_level_db = config['min_level_db'])

                print('done norm clip')

            # Every 50 files i create a subfolder, its a specific google colab need to run without crashing
            if(i%250 == 0):
                actual_store_folder = os.path.join(outdir,f'train/{i}')
                prepare_directory(actual_store_folder)
            
            if(log10mel.shape[0] < config['max_seq_mel']):
                # Thats a specific string treatment to get the file name, it needs because of "\" url dir
                mel_path = os.path.join(actual_store_folder, train_df.wav_path.values[i].rsplit('/', 1)[-1][:-4] + '.npy') 
                mel_path = mel_path.replace('\\', '/') 
                mel_dirs.append(mel_path)
                np.save(mel_path , log10mel)

                # Saving statistics
                if(config['mel_clip_normalize'] == False):
                    spk = train_df.emb_id.values[i]
                    if(spk not in counts):
                        counts[spk] = 0
                        feat_shape = log10mel.shape[1:]
                        sum_feats[spk] = np.zeros(feat_shape, dtype=np.float64)
                        square_sum_feats[spk] = np.zeros(feat_shape, dtype=np.float64)

                    counts[spk] += log10mel.shape[0]
                    sum_feats[spk] += log10mel.sum(axis=0)
                    square_sum_feats[spk] += (log10mel ** 2).sum(axis=0)



            else:
                mel_dirs.append('ERROR_LIMITED_SIZE_ABOVE')
        
        except:
            print('error')

    # Consolidating statistics for training set
    if(config['mel_clip_normalize'] == False):
        stats = {}
        for spk in counts:
            N = counts[spk]
            mean = sum_feats[spk]/N
            std = np.sqrt(square_sum_feats[spk] / N - mean**2)

            stats_ = np.empty((2, config['n_mel_channels']), dtype = np.float64)
            stats_[0,:] = mean 
            stats_[1,:] = std
            
            stats[spk] = stats_

        # Saving as a binary pickle file in exp/conf dir
        stats_path = training_files.rsplit('/', 1)[-2] + '/stats.pkl'
        file = open(stats_path, 'wb')
        pickle.dump(stats, file)
        file.close()

        print(f"Normalization statistics of data saved in {stats_path}.")


    train_df['mel_dirs'] = mel_dirs
    train_df['durations'] = durations
    train_df.to_csv(training_files.rsplit('/', 1)[-2] + '/train_df_with_mels.csv', index = False)
    print(f"Training features created. Auxiliar dataframe created in {validation_files.rsplit('/',1)[-2]}")

    ################################ VALIDATION PROCESS ############################################

    actual_store_folder = os.path.join(outdir,'val/0')
    prepare_directory(actual_store_folder)

    # List that will store mel paths
    mel_dirs = []
    durations = []

    # Starting Validation files process
    print(f"Saving validation features in {os.path.join(outdir, 'val')} folder.")
    for i in range(val_df.shape[0]):

        try: 
            audio , sr = librosa.load(val_df.wav_path.values[i], sr = config['sampling_rate'])

            # Trimming silence
            audio = librosa.effects.trim(audio, top_db=config['trim_treshold_in_db'], 
                                        frame_length=config['trim_frame_size'],
                                        hop_length=config['trim_hop_size'])[0]
            audio = np.append([0.]*3*config['trim_hop_size'], audio)
            audio = np.append(audio, [0.]*3*config['trim_hop_size'])

            durations.append(len(audio)/sr)

            log10mel = logmelfilterbank(audio, config['sampling_rate'], config['filter_length'], 
                                    config['hop_length'], config['win_length'], config['window'],
                                    config['n_mel_channels'], config['mel_fmin'], config['mel_fmax'])

            if(config['mel_clip_normalize'] == True):
                log10mel = mel_normalize(log10mel, ref_level_db = config['ref_level_db'], 
                                        max_abs_value = config['max_abs_value'], min_level_db = config['min_level_db'])            

            # Every 50 files i create a subfolder, its a specific google colab needed to run without crashing
            if(i%50 == 0):
                actual_store_folder = os.path.join(outdir,f'val/{i}')
                prepare_directory(actual_store_folder)
            
            if(log10mel.shape[0] < config['max_seq_mel']):
                # Thats a specific string treatment to get the file name, it needs because of "\" url dir
                mel_path = os.path.join(actual_store_folder, val_df.wav_path.values[i].rsplit('/', 1)[-1][:-4] + '.npy')
                mel_path = mel_path.replace('\\', '/') 
                mel_dirs.append(mel_path)
                np.save(mel_path , log10mel)
            else:
                mel_dirs.append('ERROR_LIMITED_SIZE_ABOVE')
    
        except:
            print('error')

    val_df['mel_dirs'] = mel_dirs
    val_df['durations'] = durations
    val_df.to_csv(validation_files.rsplit('/', 1)[-2] + '/val_df_with_mels.csv', index = False)
    print(f"Validation features created. Auxiliar dataframe created in {validation_files.rsplit('/',1)[-2]}")



if __name__ == '__main__':
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py).")
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save mel files')
    parser.add_argument('-c', '--conf', type=str,
                        help='directory where the yaml config file is')
    args = parser.parse_args()

    with open(args.conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # config.update(vars(args))
    main(args.output_directory, config)
    