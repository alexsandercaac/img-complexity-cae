"""
    In this stage, the reconstruction error of the CAE is used as am outlier
    score and prediction is made based on a threshold defined with the
    validation data.
"""
import pandas as pd
from utils.models.scikit_cae_estimator import ScikitCaeEstimator
from skopt import BayesSearchCV
from skopt.space import Real

cae_df = pd.read_csv('data/processed/tabular/cae_mse.csv')

cae_df['label'] = cae_df['label'].apply(
    lambda x: 1 if x == 'def_front' else 0)

mask = cae_df['data_split'] == 'train'
train_df = cae_df[mask].drop(columns=['data_split'])
mask = cae_df['data_split'] == 'val'
val_df = cae_df[mask].drop(columns=['data_split'])
mask = cae_df['data_split'] == 'test'
test_df = cae_df[mask].drop(columns=['data_split'])


params_grid = {
    'threshold': Real(
        val_df['cae_mse'].min(), val_df['cae_mse'].max(), prior='uniform')
}

cv_splits = (
    ([0], list(range(len(val_df)))) for _ in range(1)
)

model = ScikitCaeEstimator()

clf = BayesSearchCV(model, params_grid, n_jobs=-1, cv=cv_splits, n_iter=25)

search = clf.fit(val_df['cae_mse'].values, val_df['label'].values)

best_threshold = search.best_params_['threshold']
best_score = search.best_score_