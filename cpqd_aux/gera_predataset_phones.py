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
    args = parser.parse_args()

    cpqd_path = args.input_directory

    wav_dirs = []
    texts = []
    emb_ids = []
    style_targets = []

    for file in os.listdir(cpqd_path):

        if(os.path.isdir(os.path.join(cpqd_path,file))):

            folders_path = os.path.join(cpqd_path,file)
            

            if((os.path.isdir(os.path.join(folders_path,'transcricao'))) & (os.path.isdir(os.path.join(folders_path,'wav16')))):
                print('entrou no if')   

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
                            if((os.path.isfile(expected_wav_file))):
                                texts.append(line[N+2:])
                                wav_dirs.append(expected_wav_file) 
                                emb_ids.append(int(args.speaker_id)) # Since we dont have embedding just put that to generate correct format
                                print("antes do librosa")
                                x , sr = librosa.load(expected_wav_file, sr = None)
                                print('antes do arquivo lab')
                                with open(expected_lab_file , 'r') as f:
                                    for k in f.readlines():
                                        if("phones" in line[:10]):
                                            qtde_phones = len(k[10:].replace('|', '').split(), len(line[10:].replace('|', '').split()))
                                print(len(x), sr, qtde_phones)
                                style_targets.append(int(qtde_phones/len(x)/sr))
                except:
                    print('deu except')
                    pass

    df = pd.DataFrame({'wav_path':wav_dirs, 'text': texts, 'emb_id': np.array(emb_ids).astype(int), 'style_target': style_targets})


    df_train, df_val, speaker_id, speaker_id = train_test_split(df, df['emb_id'], test_size = 0.1, random_state = 42, stratify = df['emb_id'])

    out_train = os.path.join(args.output_directory, args.df_name + '_train.csv')
    out_val = os.path.join(args.output_directory, args.df_name + 'val.csv')

    df_train.to_csv(out_train, index = False, sep = '|', encoding='latin-1')
    df_val.to_csv(out_val, index = False, sep='|', encoding = 'latin-1')