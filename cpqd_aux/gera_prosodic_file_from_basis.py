'''

    Script que mapeia as pastas da Rosana das bases do CPQD, busca pelos arquivos .wav em 16Khz
    le as transcrições normalizadas (futuramente trabalharemos com a transcrição fonética) e gera os datasets da
    necessários para processarmos os áudios para testar o tacotron2.

'''

import os
import argparse 
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import pyworld as pw
import re


def get_pitch_range(audio_path, sr = None, pmin = 5, pmax = 95):
    '''
        Takes the audio (.wav) file path and return the pitch range.
        
        Here, pitch range is defined as follows in https://arxiv.org/pdf/2009.06775v1.pdf
        
        pitch range = P95(pitch) - P05(pitch), where pitch is the pitch contour ignoring silence.
        
    '''
    x, fs = librosa.load(audio_path, sr=sr)
    _f0, t = pw.dio(x.astype(np.double), fs)    # raw pitch extractor
    f0 = pw.stonemask(x.astype(np.double), _f0, t, fs)  # pitch refinement
    
    lower_bound = np.percentile(f0[f0>0], pmin)
    upper_bound = np.percentile(f0[f0>0], pmax)
    
    pitch_range = upper_bound - lower_bound
    
    return pitch_range

def get_logpitch_mean(audio_path, sr = None):
    '''
        Takes the audio (.wav) file path and return the pitch mean.
        
        Here, log pitch mean is defined as follows in https://arxiv.org/pdf/2009.06775v1.pdf
        
        log pitch mean = mean(log(pitch)), where pitch is the pitch contour ignoring silence.
        
    '''
    x, fs = librosa.load(audio_path, sr=sr)
    _f0, t = pw.dio(x.astype(np.double), fs)    # raw pitch extractor
    f0 = pw.stonemask(x.astype(np.double), _f0, t, fs)  # pitch refinement
    
    logpitch_mean = np.log(f0[f0>0]).mean()
    
    return logpitch_mean

def get_energy(audio_path, sr = None, top_level_db = 10, frame_length=1024, hop_length = 512):
    '''
        Takes the audio (.wav) file path and return the mean speech energy.
        
        Here, speech energy is defined as follows in https://arxiv.org/pdf/2009.06775v1.pdf
        
        E = 20*log(mean(abs(x))), where x is audio amplitudes without silence
        
    '''
    
    x, fs = librosa.load(audio_path, sr=sr)
    
    # Getting the non silent partitions
    non_silent_partitions = librosa.effects.split(x, top_db=top_level_db, frame_length=frame_length, hop_length=hop_length)
    x_clean = []
    for interval in non_silent_partitions:
        x_clean.extend(x[interval[0]:interval[1]])

    x_clean = np.array(x_clean) 
    
    energy = 20*np.log(abs(x).mean())
    
    return energy

def get_cpqd_lab_speaking_rate(audio_file, lab_path):
    '''
        Takes the audio (.wav) file path and lab (.lab) file path from CPqD environment.
        
        It counts the phones/duration. Which is a proxy for speaking rate.
    '''
    
    x , sr = librosa.load(audio_file, sr = None)
    
    with open(lab_path , 'r', encoding = 'latin-1') as f:
        for k in f.readlines():
            if("phones" in k[:10]):
                qtde_phones = len(k[10:].replace('|', '').split())
                break
    
    qtde_phones = round(qtde_phones/(len(x)/sr), 3)
    
    return qtde_phones

if __name__ == '__main__':
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py).")
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to auxiliar datasets')
    parser.add_argument('-i', '--input_directory', type=str,
                        help='directory of all folders of Rosana data')
    parser.add_argument('-t', '--text', type=str,
                        help='Whether to get norm text or phoneme level')
    parser.add_argument('-dn', '--df_name', type = str)
    parser.add_argument('-s', '--speaker_id', type = str)
    parser.add_argument('-sn', '--style_name', type = str, default=None)
    parser.add_argument("--ignore", nargs="+", default=["debug"])
    args = parser.parse_args()

    # cpqd_path = args.input_directory

    train_pitch_range = []
    train_speaking_rate = []
    train_energy = []

    val_pitch_range = []
    val_speaking_rate = []
    val_energy = []

    total_time = 0

    train_file = args.input_directory + '_train.csv'
    val_file = args.input_directory + '_val.csv'

    train = pd.read_csv(train_file, sep = '|', encoding = 'latin-1')
    val = pd.read_csv(val_file, sep = '|', encoding = 'latin-1')

    train_wav_dirs = []
    val_wav_dirs = []

    train_texts = []
    val_texts = []

    train_emb = []
    val_emb = []

    train_style = []
    val_style = []

    # train_wav_dirs = train['wav_path']
    # val_wav_dirs = val['wav_path']

    # train_texts = train['text']
    # val_texts = val['text']

    # train_emb = train['emb_id']
    # val_emb = val['emb_id']

    # train_style = train['style_target']
    # val_style = val['style_target']

    # Getting prosodic features

    for i, wav in enumerate(train['wav_path'].values):
        expected_wav_file = wav
        expected_lab_file = wav[:-4] + '.lab'

        try:
            if(os.path.isfile(expected_wav_file)):                    
                # train_speaking_rate.append(get_cpqd_lab_speaking_rate(expected_wav_file, expected_lab_file))                
                train_pitch_range.append(get_pitch_range(expected_wav_file))
                # train_energy.append(get_energy(expected_wav_file, sr = None, top_level_db=15, frame_length = 512, hop_length=128))
                
                train_texts.append(train['text'].values[i])
                train_wav_dirs.append(expected_wav_file) 
                train_emb.append(train['emb_id'].values[i]) # Since we dont have embedding just put that to generate correct format
                train_style.append(train['style_target'].values[i])
        except:
            print('deu ruim train')

        
    for i, wav in enumerate(val['wav_path'].values):
        expected_wav_file = wav
        expected_lab_file = wav[:-4] + '.lab'
        print(expected_lab_file)

        try:
            if(os.path.isfile(expected_wav_file)):                 
                # val_speaking_rate.append(get_cpqd_lab_speaking_rate(expected_wav_file, expected_lab_file))  
                # print('passou spk')              
                val_pitch_range.append(get_pitch_range(expected_wav_file))
                print('passou pich')
                # val_energy.append(get_energy(expected_wav_file, sr = None, top_level_db=15, frame_length = 512, hop_length=128))
                # print('passou energy')

                val_texts.append(val['text'].values[i])
                val_wav_dirs.append(expected_wav_file) 
                val_emb.append(val['emb_id'].values[i]) # Since we dont have embedding just put that to generate correct format
                val_style.append(val['style_target'].values[i])
        except:
            print('deu ruim val')



    print(len(train_pitch_range))
    print(len(val_pitch_range))

    # Normalizing prosodic values
    # mean_speaking_rate = np.mean(np.array(train_speaking_rate))
    # std_speaking_rate = np.std(np.array(train_speaking_rate))

    # train_speaking_rate = np.clip((np.array(train_speaking_rate) - mean_speaking_rate)/std_speaking_rate, -1, 1)
    # val_speaking_rate = np.clip((np.array(val_speaking_rate) - mean_speaking_rate)/std_speaking_rate, -1, 1)

    mean_pitch_range = np.mean(np.array(train_pitch_range))
    std_pitch_range = np.std(np.array(train_pitch_range))

    train_pitch_range = np.clip((np.array(train_pitch_range) - mean_pitch_range)/std_pitch_range, -1, 1)
    val_pitch_range = np.clip((np.array(val_pitch_range) - mean_pitch_range)/std_pitch_range, -1, 1)

    # mean_energy = np.mean(np.array(train_energy))
    # std_energy = np.std(np.array(train_energy))

    # train_energy = np.clip((np.array(train_energy) - mean_energy)/std_energy, -1, 1)
    # val_energy = np.clip((np.array(val_energy) - mean_energy)/std_energy, -1, 1)

    with open('/'.join(args.input_directory.split('/')[:-1]) + 'prosodic_stat.txt', 'w') as f:
        f.write(f'mean pitch range: {mean_pitch_range}')
        # f.write(f'mean speaking_rate: {mean_speaking_rate}')
        # f.write(f'mean energy: {mean_energy}')

    # print(len(wav_dirs), len(texts), len(emb_ids), len(style_targets), total_time/3600)
    # df_train = pd.DataFrame({'wav_path': train_wav_dirs, 'text': train_texts, 'emb_id': train_emb, 'style_target': train_style, 'pitch_range': train_pitch_range, 'energy': train_energy, 'speaking_rate': train_speaking_rate})
    # df_val = pd.DataFrame({'wav_path': val_wav_dirs, 'text': val_texts, 'emb_id': val_emb, 'style_target': val_style, 'pitch_range': val_pitch_range, 'energy': val_energy, 'speaking_rate': val_speaking_rate})
    df_train = pd.DataFrame({'wav_path': train_wav_dirs, 'text': train_texts, 'emb_id': train_emb, 'style_target': train_style, 'pitch_range': train_pitch_range})
    df_val = pd.DataFrame({'wav_path': val_wav_dirs, 'text': val_texts, 'emb_id': val_emb, 'style_target': val_style, 'pitch_range': val_pitch_range})
    



    # df['text'] = df['text'].str.lower().str.replace('.', ' .').str.replace(',', ' ,').str.replace('!', ' !').str.replace('?', ' ?').str.replace('\n', '').str.replace('\t', '').str.replace('''"''', '')

    # df_train, df_val, _, _ = train_test_split(df, df['emb_id'], test_size = 0.1, random_state = 42, stratify = df['emb_id'])

    out_train = os.path.join(args.output_directory, args.df_name + '_train.csv')
    out_val = os.path.join(args.output_directory, args.df_name + '_val.csv')

    df_train.to_csv(out_train, index = False, sep = '|', encoding='latin-1')
    df_val.to_csv(out_val, index = False, sep='|', encoding = 'latin-1')