import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / 'data' / 'processed'

df_train = pd.read_csv(data_dir / 'train.csv')
df_val = pd.read_csv(data_dir / 'val.csv')

y_train = pd.read_csv(data_dir / 'y_train.csv').values.ravel()
y_val = pd.read_csv(data_dir / 'y_val.csv').values.ravel()

print(df_train.shape)
print(df_val.shape)
print(y_train.shape)
print(y_val.shape)

df_full_train = pd.concat([df_train, df_val], ignore_index=True)
y_full_train = np.concatenate([y_train, y_val])