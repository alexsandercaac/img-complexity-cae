"""
    In this stage, plots with image complexity data are generated.
"""
import pandas as pd


complexity_df = pd.read_csv('../data/processed/tabular/complexity.csv')

mask = complexity_df['data_split'] == 'train'
complexity_df = complexity_df[mask].drop(columns=['data_split'])
