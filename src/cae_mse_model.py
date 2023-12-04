"""
    In this stage, the reconstruction error of the CAE is used as am outlier
    score and prediction is made based on a threshold defined with the
    validation data.
"""
import os

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from utils.misc import create_dir
from utils.dvc.params import get_params


params = get_params('all')

SEED = params['seed']
DATASET = params['dataset']
MODEL_DIR = os.path.join('models', DATASET, 'bin')
create_dir(MODEL_DIR)

# Algorithm parameters
SCORE_FUNC = params['models_score_func']
N_ITER = params['models_n_iter']
N_SPLITS = params['models_n_splits']

cae_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'cae_mse.csv'),
    index_col=0
)

cae_df['label'] = cae_df['label'].apply(
    lambda x: 1 if x == 'positive' else 0)

# We will be using cross validation, so we can merge the train and validation
mask = (cae_df['data_split'] == 'val') | (cae_df['data_split'] == 'train')
train_df = cae_df[mask].drop(columns=['data_split'])
class_weights = {
    0: len(train_df) / (2 * train_df['label'].value_counts()[0]),
    1: len(train_df) / (2 * train_df['label'].value_counts()[1])
}

grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_ITER, random_state=SEED)

grid_search = GridSearchCV(
    estimator=LogisticRegression(
        class_weight=class_weights,
        random_state=SEED),
    param_grid=grid,
    scoring=SCORE_FUNC,
    cv=cv,
    n_jobs=-1,
    verbose=1
)

inputs = train_df.drop(columns=['label'])
targets = train_df['label']

search_results = grid_search.fit(inputs, targets)

means = search_results.cv_results_['mean_test_score']
stds = search_results.cv_results_['std_test_score']
params = search_results.cv_results_['params']
print("Grid scores:")
for mean, stdev, param in zip(means, stds, params):
    print(f"\nScore: {mean:.3f} ({stdev:.3f}),\nWith params: {param}")

print("\n==========================\n")

print(
    f"Found best score of {search_results.best_score_}" +
    f" with params: {search_results.best_params_}")


# Train model with best params

model = LogisticRegression(
    **search_results.best_params_,
    class_weight=class_weights,
    random_state=SEED
)

model = model.fit(inputs, targets)

# Save model
model_path = os.path.join(MODEL_DIR, 'logistic_regression_cae.joblib')
dump(model, model_path)
