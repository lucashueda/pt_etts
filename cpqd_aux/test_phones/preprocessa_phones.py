import pandas as pd 


df_train = pd.read_csv('./rosana_train.csv', encoding='latin-1', sep = '|')

df_val = pd.read_csv('./rosana_val.csv', encoding='latin-1', sep = '|')


inter1 = 8.927
inter2 = 11.5979

map_t = {1: 't_lento',
        2: 't_medio',
        3: 't_rapido'}

df_train[df_train.style_target < inter1]['style_target'] = 1
df_train[(df_train.style_target >= inter1) & (df_train.style_target < inter2)]['style_target'] = 2
df_train[df_train.style_target >= inter2]['style_target'] = 3

df_val[df_val.style_target < inter1]['style_target'] = 1
df_val[(df_val.style_target >= inter1) & (df_val.style_target < inter2)]['style_target'] = 2
df_val[df_val.style_target >= inter2]['style_target'] = 3

df_train['style_target'] = df_train['style_target'].map(map_t)
df_val['style_target'] = df_val['style_target'].map(map_t)

print(df_train.style_target.unique(), df_val.style_target.unique())

df_train.to_csv('rosana_train.csv', encoding='latin-1', index=False, sep = '|')
df_val.to_csv('rosana_val.csv', encoding='latin-1', index=False, sep = '|')