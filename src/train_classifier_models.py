"""
    In this stage, the CAE reconstruction MSE and JPEG compression MSE are
    used to train a binary classifier to predict the image label.
"""
import os

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from joblib import dump
import pandas as pd

from utils.dvc.params import get_params
from utils.misc import create_dir


params = get_params()

SEED = params['seed']
DATASET = params['dataset']
MODEL_DIR = os.path.join('models', DATASET, 'bin')
create_dir(MODEL_DIR)
INPUT_PATH = os.path.join(
    'data', 'processed', DATASET, 'tabular', 'complexity_caemse.csv')

# Algorithm parameters
SCORE_FUNC = params['models_score_func']
N_ITER = params['models_n_iter']
N_SPLITS = params['models_n_splits']

cae_df = pd.read_csv(
    INPUT_PATH,
    index_col=0
)

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

features = {
    'complexity': ['jpeg_mse', 'delentropy'],
    'autoencoder': ['cae_mse'],
    'combined': ['jpeg_mse', 'delentropy', 'cae_mse']
}

for model_type, cols in features.items():
    print("\n==========================\n")
    print(f"Training model with {model_type} features")
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

    inputs = train_df[cols]
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
    model_path = os.path.join(MODEL_DIR, 'logistic_regression_' +
                              model_type + '.joblib')
    dump(model, model_path)
