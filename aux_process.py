import pandas as pd
import os 
# Lendo os arquivos .npy

mels_path = './mel_dir/'

mels_dir = []
wav_ids = []
for file in os.listdir(mels_path):
  mel_files = os.path.join(mels_path, file)
  for part in os.listdir(mel_files):
    part_files = os.path.join(mel_files,part)
    for mel in os.listdir(part_files):
      mels_dir.append(os.path.join(part_files, mel))
      wav_ids.append(mel[:-4])
print(len(mels_dir))

df_final = pd.read_csv('./wav_txt_pairs.csv')

df_final['text_n'] = df_final.texts.str.replace('\n', '')

df_mels = pd.DataFrame({'mel_path': mels_dir, 'id': wav_ids})

df_final = df_final.merge(df_mels, how= 'left', on = 'id')

# limpando txt_path nulos
df_final = df_final[~df_final.mel_path.isnull()]

df_final = df_final.drop_duplicates(subset='mel_path')

df_final = pd.read_csv('./mel_txt_pairs.csv')

df_final['text_n'] = df_final.texts.str.replace('\n', '')

# Criando speakers
df_final['speaker'] = df_final.id.str[:4]


# Limpando arquivo com extensão corrompida
df_final['wav_ext'] = df_final.wav_path.str[-3:]
df_final = df_final[df_final['wav_ext'] == 'wav']

df_final[['mel_path','text_n','speaker']].to_csv('./mel_txt_pairs_use_withemb.txt', header=None, index = False, sep='|')

# Creating fake train and val files
df_final[['mel_path','text_n','speaker']].to_csv('./train.txt', header=None, index = False, sep='|')
df_final.sample(frac = 0.01)[['mel_path','text_n','speaker']].to_csv('./val.txt', header=None, index = False, sep='|')



