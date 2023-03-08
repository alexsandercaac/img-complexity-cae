"""
    In this stage, the reconstruction error of the CAE is used as am outlier
    score and prediction is made based on a threshold defined with the
    validation data.
"""
import pandas as pd
from sklearn.metrics import f1_score

from utils.models.threshold_search import bayesian_search_th
from utils.dvc.params import get_params


params = get_params()
cae_df = pd.read_csv('data/processed/tabular/cae_mse.csv')

cae_df['label'] = cae_df['label'].apply(
    lambda x: 1 if x == 'def_front' else 0)

mask = cae_df['data_split'] == 'train'
train_df = cae_df[mask].drop(columns=['data_split'])
mask = cae_df['data_split'] == 'val'
val_df = cae_df[mask].drop(columns=['data_split'])
mask = cae_df['data_split'] == 'test'
test_df = cae_df[mask].drop(columns=['data_split'])

if params['score_func'] == 'accuracy':
    score_func = None
elif params['score_func'] == 'f1':
    score_func = f1_score
else:
    raise ValueError(
        f"Invalid score function {params['score_func']} specified.")

search_results = bayesian_search_th(
    val_df['cae_mse'].values, val_df['label'].values,
    val_df['cae_mse'].min(), val_df['cae_mse'].max(),
    n_iter=params['n_iter'], score_func=score_func
)

# Write best threshold to file in models/params
with open('models/params/cae_threshold.txt', 'w') as f:
    f.write(str(search_results['threshold']))
