# -*- coding: latin-1 -*-
import pandas as pd 


alfredo_train = pd.read_csv('./rosana_train.csv', encoding='latin-1', sep = '|')

alfredo_val = pd.read_csv('./rosana_val.csv', encoding='latin-1', sep = '|')

alfredo_train['text'] = alfredo_train['text'].str.lower().str.replace('.', ' .').str.replace(',', ' ,').str.replace('!', ' !').str.replace('?', ' ?').str.replace('\n', '').str.replace('\t', '').str.replace('''"''', '')
alfredo_val['text'] = alfredo_val['text'].str.lower().str.replace('.', ' .').str.replace(',', ' ,').str.replace('!', ' !').str.replace('?', ' ?').str.replace('\n', '').str.replace('\t', '').str.replace('''"''', '')

alfredo_train.to_csv('rosana_train.csv', encoding='latin-1', index=False, sep = '|')
alfredo_val.to_csv('rosana_val.csv', encoding='latin-1', index=False, sep = '|')