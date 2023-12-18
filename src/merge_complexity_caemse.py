"""
    In this stage, plots with image complexity data are generated.
"""
import os

import pandas as pd

from utils.dvc.params import get_params


params = get_params('all')

DATASET = params['dataset']


complexity_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'complexity.csv'),
    dtype={'file': str, 'label': str, 'data_split': str,
           'jpeg_mse': float, 'delentropy': float})

cae_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'cae_mse.csv'),
    dtype={'file': str, 'label': str, 'data_split': str,
           'cae_mse': float})

complexity_df = complexity_df.drop(columns=['data_split', 'label'])

complexity_caemse_df = complexity_df.merge(cae_df, on='file')

complexity_caemse_df['label'] = complexity_caemse_df['label'].apply(
    lambda x: 1 if x == 'positive' else 0)

complexity_caemse_df.to_csv(
    os.path.join(
        'data', 'processed', DATASET, 'tabular', 'complexity_caemse.csv'),
    index=False
)
