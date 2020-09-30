'''
  Esse arquivo consta um script consolidado para formatação dos arquivos de entrada do sistema ETTS (Expressive Text to Speech)

  Vale ressaltar que o código é feito para alguns exemplos de arquivos utilizados em experimentos, caso se queira utilizar para outras bases
  de dados é necessário adicionar algum módulo que processe os diretórios da base. Um exemplo de como do fluxograma do script é apresentado
  em "CUSTOM_PROCESS.md" no repositório.

  License: GNU V3
  Lucas Hideki Ueda, 2020.

'''

# Import de libs
import argparse
import os
import time
import pandas as pd 
import numpy as np 
import librosa
from hparams import create_hparams

# Definindo funções

def prepare_directory(out_dir):
  '''
    Função que cria um diretório de saída de arquivos
  '''
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
    os.chmod(out_dir, 0o775) # Não tenho certeza se funciona no windows

def np_wav2mel(wav_filepath, n_fft = 2048, hop_length = 512, n_mels = 80, n_pad = 200):
  '''
    Função que recebe um diretório de um arquivo .wav e retorna o melespectrograma como um vetor do numpy.

    Essa função usa a librosa para a conversão e a numpy como vetor de saída.
  
    Shape track: 
      L -> Tamanho da sequência
      D -> Dimensão do espectrograma
  '''

  data, sampling_rate = librosa.load(wav_filepath)

  mel_spec = librosa.feature.melspectrogram(data, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) # (D, L)

  return mel_spec.T # (L, D)

def generate_mel_files(in_dir, out_dir, hparams, df = 'VCTK'):
  '''
    Função que recebe o diretório dos arquivos .wav, o diretório do arquivo de output e qual o dataset trabalhado

    A localização do in_dir é totalmente relacionada com o df selecionado. É necessário editar essa função para trabalhar com 
    datasets diferentes

  '''

  # Criando o diretório onde será salvo os mels
  prepare_directory(out_dir)

  if(df == 'VCTK'):
    vctk_path = in_dir


    text_dirs = []
    wav_dirs = []
    mel_dir = []

    text_id = []
    wav_id = []

    # Vamos percorrer as pastas do diretório da base vctk
    for file in os.listdir(vctk_path):
      if(file =='wav48'):
        wav_files = os.path.join(vctk_path, file)
        for folder in os.listdir(wav_files):
          wav_folder_files = os.path.join(wav_files, folder)
          for wav in os.listdir(wav_folder_files):
            wav_dirs.append(os.path.join(wav_folder_files, wav))
            wav_id.append(wav[:-4])

            mel = np_wav2mel(os.path.join(wav_folder_files, wav))
            mel_out_path = os.path.join(out_dir, wav_id)
            np.save(mel_out_path + '.npy', mel)
            mel_dir.append(mel_out_path)


      if(file == 'txt'):
        txt_files = os.path.join(vctk_path, file)
        for folder in os.listdir(txt_files):
          txt_folder_files = os.path.join(txt_files, folder)
          for txt in os.listdir(txt_folder_files):
            text_dirs.append(os.path.join(txt_folder_files, txt))
            text_id.append(txt[:-4])

    
    df_texts = pd.DataFrame({'txt_path': text_dirs, 'id': text_id})
    df_wavs = pd.DataFrame({'wav_path': wav_dirs, 'id': wav_id, 'mel_path': mel_dir})

    print("Starting sanity check...")

    # Juntando os dataframes de texto e audios
    df_final = df_wavs.merge(df_texts, how= 'left', on = 'id')

    # Printando alguns dados da base
    print(f"{df_final[df_final.txt_path.isnull()].shape[0]} caminhos de texto são nulos e {df_final[df_final.id.isnull()].shape[0]} ids são nulos")

    # Limpando casos onde não há texto pareado
    df_final = df_final[~df_final.txt_path.isnull()]

    # Limpando audios duplicados
    df_final = df_final.drop_duplicates(subset='wav_path')

    # Normalizando os textos, tirando o '\n' do final
    df_final['text_n'] = df_final.texts.str.replace('\n', '') 

    # Especificando o speaker id
    df_final['speaker'] = df_final.id.str[:4]

    # Checando consistencia da extensão dos arquivos .wav
    df_final['wav_ext'] = df_final.wav_path.str[-3:]
    
    print(f"{df_final[df_final['wav_ext'] != 'wav'].shape[0]} observações possuem extensão quebrada.")
    
    df_final = df_final[df_final['wav_ext'] == 'wav']

    # Printando resultados finais
    print("Processo finalizado... O arquivo <lookup_table.txt> foi criado no diretório relativo atual!")

    df_final[['mel_path','text_n','speaker']].to_csv('lookup_table.txt', header=None, index = False, sep='|')


# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save mel files')
    parser.add_argument('-i', '--input_directory', type=str,
                        help='directory where the wav files will be there')

    args = parser.parse_args()

    hparams = create_hparams()

    generate_mel_files(args.input_directory, args.output_directory, hparams, df = 'VCTK')
