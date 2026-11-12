import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / 'data' / 'processed'

df_train = pd.read_csv(data_dir / 'train.csv', index_col=0)
df_val = pd.read_csv(data_dir / 'val.csv', index_col=0)

print(df_train.columns.tolist())
print('Unnamed: 0' in df_train.columns)

y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
y_val = pd.read_csv(data_dir / 'y_val.csv').values.ravel()

print(df_train.shape)
print(df_val.shape)
print(y_train.shape)
print(y_val.shape)

df_full_train = pd.concat([df_train, df_val], ignore_index=True)
y_full_train = np.concatenate([y_train, y_val])

dv = DictVectorizer(sparse=False)

train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(train_dicts)

