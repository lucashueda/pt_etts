# -*- coding: latin-1 -*-
import os
from glob import glob
import re
import sys
from pathlib import Path

from tqdm import tqdm

from TTS.tts.utils.generic_utils import split_dataset


def load_meta_data(datasets):
    meta_data_train_all = []
    meta_data_eval_all = []
    for dataset in datasets:
        name = dataset['name']
        root_path = dataset['path']
        meta_file_train = dataset['meta_file_train']
        meta_file_val = dataset['meta_file_val']
        preprocessor = get_preprocessor_by_name(name)
        meta_data_train = preprocessor(root_path, meta_file_train)
        print(f" | > Found {len(meta_data_train)} files in {Path(root_path).resolve()}")
        if meta_file_val is None:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
        else:
            meta_data_eval = preprocessor(root_path, meta_file_val)
        meta_data_train_all += meta_data_train
        meta_data_eval_all += meta_data_eval
    return meta_data_train_all, meta_data_eval_all


def get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def tweb(root_path, meta_file):
    """Normalize TWEB dataset.
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "tweb"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('\t')
            wav_file = os.path.join(root_path, cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items


# def kusal(root_path, meta_file):
#     txt_file = os.path.join(root_path, meta_file)
#     texts = []
#     wavs = []
#     with open(txt_file, "r", encoding="utf8") as f:
#         frames = [
#             line.split('\t') for line in f
#             if line.split('\t')[0] in self.wav_files_dict.keys()
#         ]
#     # TODO: code the rest
#     return  {'text': texts, 'wavs': wavs}


def mozilla(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append([text, wav_file, speaker_name])
    return items


def mozilla_de(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r', encoding="ISO 8859-1") as ttf:
        for line in ttf:
            cols = line.strip().split('|')
            wav_file = cols[0].strip()
            text = cols[1].strip()
            folder_name = f"BATCH_{wav_file.split('_')[0]}_FINAL"
            wav_file = os.path.join(root_path, folder_name, wav_file)
            items.append([text, wav_file, speaker_name])
    return items


def mailabs(root_path, meta_files=None):
    """Normalizes M-AI-Labs meta data files to TTS format"""
    speaker_regex = re.compile(
        "by_book/(male|female)/(?P<speaker_name>[^/]+)/")
    if meta_files is None:
        csv_files = glob(root_path + "/**/metadata.csv", recursive=True)
    else:
        csv_files = meta_files
    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for csv_file in csv_files:
        txt_file = os.path.join(root_path, csv_file)
        folder = os.path.dirname(txt_file)
        # determine speaker based on folder structure...
        speaker_name_match = speaker_regex.search(txt_file)
        if speaker_name_match is None:
            continue
        speaker_name = speaker_name_match.group("speaker_name")
        print(" | > {}".format(csv_file))
        with open(txt_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                if meta_files is None:
                    wav_file = os.path.join(folder, 'wavs', cols[0] + '.wav')
                else:
                    wav_file = os.path.join(root_path,
                                            folder.replace("metadata.csv", ""),
                                            'wavs', cols[0] + '.wav')
                if os.path.isfile(wav_file):
                    text = cols[1].strip()
                    items.append([text, wav_file, speaker_name])
                else:
                    raise RuntimeError("> File %s does not exist!" %
                                       (wav_file))
    return items


def ljspeech(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items


def nancy(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "nancy"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            utt_id = line.split()[1]
            text = line[line.find('"') + 1:line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", utt_id + ".wav")
            items.append([text, wav_file, speaker_name])
    return items


def common_voice(root_path, meta_file):
    """Normalize the common voice meta data file to TTS format."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            wav_file = os.path.join(root_path, "clips", cols[1].replace(".mp3", ".wav"))
            items.append([text, wav_file, 'MCV_' + speaker_name])
    return items


def libri_tts(root_path, meta_files=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if meta_files is None:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split('.')[0]
        speaker_name = _meta_file.split('_')[0]
        chapter_id = _meta_file.split('_')[1]
        _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
        with open(meta_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('\t')
                wav_file = os.path.join(_root_path, cols[0] + '.wav')
                text = cols[1]
                items.append([text, wav_file, 'LTTS_' + speaker_name])
    for item in items:
        assert os.path.exists(
            item[1]), f" [!] wav files don't exist - {item[1]}"
    return items


def custom_turkish(root_path, meta_file):
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "turkish-female"
    skipped_files = []
    with open(txt_file, 'r', encoding='utf-8') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs',
                                    cols[0].strip() + '.wav')
            if not os.path.exists(wav_file):
                skipped_files.append(wav_file)
                continue
            text = cols[1].strip()
            items.append([text, wav_file, speaker_name])
    print(f" [!] {len(skipped_files)} files skipped. They don't exist...")
    return items


# ToDo: add the dataset link when the dataset is released publicly
def brspeech(root_path, meta_file):
    '''BRSpeech 3.0 beta'''
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            if line.startswith("wav_filename"):
                continue
            cols = line.split('|')
            #print(cols)
            wav_file = os.path.join(root_path, cols[0])
            text = cols[2]
            speaker_name = cols[3]
            items.append([text, wav_file, speaker_name])
    return items


def vctk(root_path, meta_files=None, wavs_path='wav48'):
    """homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"""
    test_speakers = meta_files
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file,
                                                  root_path).split(os.sep)
        file_id = txt_file.split('.')[0]
        if isinstance(test_speakers,
                      list):  # if is list ignore this speakers ids
            if speaker_id in test_speakers:
                continue
        with open(meta_file) as file_text:
            text = file_text.readlines()[0]
        wav_file = os.path.join(root_path, wavs_path, speaker_id,
                                file_id + '.wav')
        items.append([text, wav_file, 'VCTK_' + speaker_id])

    return items


def vctk_slim(root_path, meta_files=None, wavs_path='wav48'):
    """homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"""
    items = []
    txt_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for text_file in txt_files:
        _, speaker_id, txt_file = os.path.relpath(text_file,
                                                  root_path).split(os.sep)
        file_id = txt_file.split('.')[0]
        if isinstance(meta_files, list):  # if is list ignore this speakers ids
            if speaker_id in meta_files:
                continue
        wav_file = os.path.join(root_path, wavs_path, speaker_id,
                                file_id + '.wav')
        items.append([None, wav_file, 'VCTK_' + speaker_id])

    return items

# ======================================== VOX CELEB ===========================================
def voxceleb2(root_path, meta_file=None):
    """
    :param meta_file   Used only for consistency with load_meta_data api
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="2")


def voxceleb1(root_path, meta_file=None):
    """
    :param meta_file   Used only for consistency with load_meta_data api
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="1")


def _voxcel_x(root_path, meta_file, voxcel_idx):
    assert voxcel_idx in ["1", "2"]
    expected_count = 148_000 if voxcel_idx == "1" else 1_000_000
    voxceleb_path = Path(root_path)
    cache_to = voxceleb_path / f"metafile_voxceleb{voxcel_idx}.csv"
    cache_to.parent.mkdir(exist_ok=True)

    # if not exists meta file, crawl recursively for 'wav' files
    if meta_file is not None:
        with open(str(meta_file), 'r') as f:
            return [x.strip().split('|') for x in f.readlines()]

    elif not cache_to.exists():
        cnt = 0
        meta_data = ""
        wav_files = voxceleb_path.rglob("**/*.wav")
        for path in tqdm(wav_files, desc=f"Building VoxCeleb {voxcel_idx} Meta file ... this needs to be done only once.",
                         total=expected_count):
            speaker_id = str(Path(path).parent.parent.stem)
            assert speaker_id.startswith('id')
            text = None  # VoxCel does not provide transciptions, and they are not needed for training the SE
            meta_data += f"{text}|{path}|voxcel{voxcel_idx}_{speaker_id}\n"
            cnt += 1
        with open(str(cache_to), 'w') as f:
            f.write(meta_data)
        if cnt < expected_count:
            raise ValueError(f"Found too few instances for Voxceleb. Should be around {expected_count}, is: {cnt}")

    with open(str(cache_to), 'r') as f:
        return [x.strip().split('|') for x in f.readlines()]


def rosana(root_path, meta_file):
    meta_path = os.path.join(root_path, meta_file)
    
    items = []
    
    with open(meta_path, 'r', encoding= 'latin-1') as f:
        for line in f:
            cols = line.split('|')
            if(cols[1] == 'text'): # The first element is header in my file
                continue
            wav_file = cols[0]
            text = cols[1]
            speaker_name = 'rosana'
            items.append([text, wav_file, speaker_name])
    return items

def vctk_direct(root_path, meta_file):
    meta_path = os.path.join(root_path, meta_file)
    
    items = []
    
    with open(meta_path, 'r', encoding= 'latin-1') as f:
        for line in f:
            cols = line.split('|')
            if(cols[1] == 'text'): # The first element is header in my file
                continue
            wav_file = cols[0]
            text = cols[1]
            speaker_name = cols[2][:-1]  # The last char is "\n" since after this line is a breakline
            items.append([text, wav_file, speaker_name])
    return items

def style_reader(root_path, meta_file):
    '''
        Generic reader for files with style target

        Input file must be a textual file separated by "|" with "\n" breaklines,
        and the columns must be:

        wav_file path | text | speaker_name | style target

        The output will be:

        [text, wav_path, speaker_name, style_target]

    '''
    
    meta_path = os.path.join(root_path, meta_file)

    items = []

    with open(meta_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            cols = line.split('|')
            if(cols[1] == 'text'): # It indicates that the first row is the header so we need to skip
                continue
            wav_file = cols[0]
            text = cols[1]
            speaker_name = cols[2]
            style_target = cols[3][:-1] # The last char is always "\n" since after this line is a breakline
            items.append([text,wav_file,speaker_name,style_target])

    return items

def style_prosodic_reader(root_path, meta_file):
    '''
        Generic reader for files with style target

        Input file must be a textual file separated by "|" with "\n" breaklines,
        and the columns must be:

        wav_file path | text | speaker_name | style target

        The output will be:

        [text, wav_path, speaker_name, style_target]

    '''
    
    meta_path = os.path.join(root_path, meta_file)

    items = []

    with open(meta_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            cols = line.split('|')
            if(cols[1] == 'text'): # It indicates that the first row is the header so we need to skip
                continue
            wav_file = cols[0]
            text = cols[1]
            speaker_name = cols[2]
            style_target = cols[3]
            pitch_range = float(cols[4])
            energy = float(cols[5])
            speaking_rate = float(cols[6][:-1]) # The last char is always "\n" since after this line is a breakline
            items.append([text,wav_file,speaker_name,style_target, pitch_range, speaking_rate, energy])

    return items

def pitch_prosodic_reader(root_path, meta_file):
    '''
        Generic reader for files with style target

        Input file must be a textual file separated by "|" with "\n" breaklines,
        and the columns must be:

        wav_file path | text | speaker_name | style target

        The output will be:

        [text, wav_path, speaker_name, style_target]

    '''
    
    meta_path = os.path.join(root_path, meta_file)

    items = []

    with open(meta_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            cols = line.split('|')
            if(cols[1] == 'text'): # It indicates that the first row is the header so we need to skip
                continue
            wav_file = cols[0]
            text = cols[1]
            speaker_name = cols[2]
            style_target = cols[3]
            pitch_range = float(cols[4][:-1])
            # energy = float(cols[5])
            # speaking_rate = float(cols[6][:-1]) # The last char is always "\n" since after this line is a breakline
            items.append([text,wav_file,speaker_name,style_target, pitch_range])

    return items