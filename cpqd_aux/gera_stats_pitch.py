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
import json

if __name__ == '__main__':
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Generate stats file from a folder with csv files.")
    parser.add_argument('-fp', '--folder_path', type=str,
                        help='folder path where are the .csv files')
    # parser.add_argument('-i', '--input_directory', type=str,
    #                     help='directory of all folders of Rosana data')
    # parser.add_argument('-t', '--text', type=str,
    #                     help='Whether to get norm text or phoneme level')
    # parser.add_argument('-dn', '--df_name', type = str)
    # parser.add_argument('-s', '--speaker_id', type = str)
    # parser.add_argument('-sn', '--style_name', type = str, default=None)
    # parser.add_argument("--ignore", nargs="+", default=["debug"])
    args = parser.parse_args()


    df_full_train = pd.DataFrame()
    df_full_val = pd.DataFrame()

    for csv_file in os.listdir(args.folder_path):
        file_path = os.path.join(args.folder_path, csv_file)
        df = pd.read_csv(file_path, sep='|', encoding='latin-1')

        if('train' in csv_file):
            df_full_train = pd.concat([df_full_train, df], axis = 0, ignore_index = True)
        elif('val' in csv_file):
            df_full_val = pd.concat([df_full_val, df], axis = 0, ignore_index = True)
        else:
            print('not in format')


    stats_train = df_full_train.groupby('emb_id').agg({'pitch_range': np.mean}).reset_index()
    std_train = df_full_train.groupby('emb_id').agg({'pitch_range': np.std}).reset_index()
    stats_train['std_pitch_range'] = std_train['pitch_range']

    stats_path = os.path.join(args.folder_path, 'stats.csv')
    stats_train.to_csv(stats_path)

    # Normalizing pitch ranges
    for speaker in df_full_train.emb_id.unique():

        values_train = (df_full_train[df_full_train['emb_id'] == speaker]['pitch_range'] - stats_train[stats_train['emb_id'] == speaker]['pitch_range'].values[0])/stats_train[stats_train['emb_id'] == speaker]['std_pitch_range'].values[0]

        df_full_train.loc[df_full_train['emb_id'] == speaker, 'pitch_range'] = np.clip(values_train, -1, 1)

        
        values_val = (df_full_val[df_full_val['emb_id'] == speaker]['pitch_range'] - stats_train[stats_train['emb_id'] == speaker]['pitch_range'].values[0])/stats_train[stats_train['emb_id'] == speaker]['std_pitch_range'].values[0]

        df_full_val.loc[df_full_val['emb_id'] == speaker, 'pitch_range'] = np.clip(values_val, -1, 1)


    out_train = os.path.join(args.folder_path, 'full_df_pitch_norm_train.csv')
    out_val = os.path.join(args.folder_path, 'full_df_pitch_norm_val.csv')

    df_full_train.to_csv(out_train, index = False, sep = '|', encoding='latin-1')
    df_full_val.to_csv(out_val, index = False, sep='|', encoding = 'latin-1')