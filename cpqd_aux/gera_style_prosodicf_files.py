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

    cpqd_path = args.input_directory

    wav_dirs = []
    texts = []
    emb_ids = []
    style_targets = []
    pitch_range = []
    speaking_rate = []
    energy = []

    total_time = 0

    for file in os.listdir(cpqd_path):

        if(file not in args.ignore):
            if(os.path.isdir(os.path.join(cpqd_path,file))):

                folders_path = os.path.join(cpqd_path,file)
                

                if((os.path.isdir(os.path.join(folders_path,'transcricao'))) & (os.path.isdir(os.path.join(folders_path,'wav16')))):
                    # print('entrou no if')   

                    transcript = os.path.join(folders_path, 'transcricao')
                    wav_path = os.path.join(folders_path, 'wav16')
                    lab_path = os.path.join(folders_path, 'lab')
                    expected_norm_text_file = file+'.txt'

                    N = len(file) + 4

                    # print(transcript, wav_path, expected_norm_text_file)

                    # print(os.path.isdir(transcript), os.path.isdir(wav_path))

                    # print(os.path.isfile(os.path.join(transcript, expected_norm_text_file)))

                    try:
                        with open(os.path.join(transcript, expected_norm_text_file), encoding = 'latin-1') as f:
                            # print('entrou')
                            for line in f:
                                filename = line[:N]

                                expected_wav_file = os.path.join(wav_path, filename + '.wav')
                                expected_lab_file = os.path.join(lab_path, filename + '.lab')
                                if(os.path.isfile(expected_wav_file)):
                                    texts.append(line[N+2:])
                                    wav_dirs.append(expected_wav_file) 
                                    emb_ids.append(args.speaker_id) # Since we dont have embedding just put that to generate correct format
                                    if(args.style_name == None):
                                        style_targets.append('t_' + file[:12])
                                    else:
                                        style_targets.append(args.style_name)
                                    
                                    pitch_range.append(get_pitch_range(expected_wav_file))
                                    speaking_rate.append(get_cpqd_lab_speaking_rate(expected_wav_file, expected_lab_file))
                                    energy.append(get_energy(expected_wav_file, sr = None, top_level_db=15, frame_length = 512, hop_length=128))
                    except:
                        print('deu except')
                        pass


    # print(len(wav_dirs), len(texts), len(emb_ids), len(style_targets), total_time/3600)
    df = pd.DataFrame({'wav_path':wav_dirs, 'text': texts, 'emb_id': emb_ids, 'style_target': style_targets, 
    'pitch_range': pitch_range, 'energy': energy, 'speaking_rate': speaking_rate})
    
    df['text'] = df['text'].str.lower().str.replace('.', ' .').str.replace(',', ' ,').str.replace('!', ' !').str.replace('?', ' ?').str.replace('\n', '').str.replace('\t', '').str.replace('''"''', '')

    df_train, df_val, _, _ = train_test_split(df, df['emb_id'], test_size = 0.1, random_state = 42, stratify = df['emb_id'])

    out_train = os.path.join(args.output_directory, args.df_name + '_train.csv')
    out_val = os.path.join(args.output_directory, args.df_name + '_val.csv')

    df_train.to_csv(out_train, index = False, sep = '|', encoding='latin-1')
    df_val.to_csv(out_val, index = False, sep='|', encoding = 'latin-1')