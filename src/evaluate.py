"""
    In this stage, various classification metrics are extracted from the model.
"""
import os

import pandas as pd
import json
import joblib

from utils.evaluation.classification_metrics import get_classification_metrics
from utils.dvc.params import get_params
from utils.misc import create_dir

params = get_params('all')

DATASET = params['dataset']
MODEL_DIR = os.path.join('models', DATASET, 'bin')
METRICS_DIR = os.path.join('metrics', DATASET)
create_dir(METRICS_DIR)

cae_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'cae_mse.csv'),
    index_col=0
)

complexity_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'complexity.csv'),
    index_col=0)

complexity_df = complexity_df[['jpeg_mse']]
cae_df = cae_df.join(complexity_df)

cae_df['label'] = cae_df['label'].apply(
    lambda x: 1 if x == 'positive' else 0)

mask = cae_df['data_split'] == 'test'
test_df = cae_df[mask].drop(columns=['data_split'])

# Load models
cae_model = joblib.load(
    os.path.join(MODEL_DIR, 'logistic_regression_cae.joblib'))
jpeg_model = joblib.load(
    os.path.join(MODEL_DIR, 'logistic_regression_jpeg.joblib'))
combined_model = joblib.load(
    os.path.join(MODEL_DIR, 'logistic_regression_combined.joblib'))


cae_test_metrics = get_classification_metrics(
    test_df['label'],
    cae_model.predict(test_df.drop(columns=['label', 'jpeg_mse'])))

# Write metrics on json file
metrics = {
    'test_f1': cae_test_metrics['f1'],
    'test_acc': cae_test_metrics['accuracy'],
    'test_prec': cae_test_metrics['precision'],
    'test_rec': cae_test_metrics['recall']
}

with open(
        os.path.join(METRICS_DIR, 'cae_metrics.json'), 'w') as f:
    json.dump(metrics, f)

jpeg_test_metrics = get_classification_metrics(
    test_df['label'],
    jpeg_model.predict(test_df.drop(columns=['label', 'cae_mse'])))
metrics = {
    'test_f1': jpeg_test_metrics['f1'],
    'test_acc': jpeg_test_metrics['accuracy'],
    'test_prec': jpeg_test_metrics['precision'],
    'test_rec': jpeg_test_metrics['recall']
}

with open(
        os.path.join(METRICS_DIR, 'jpeg_metrics.json'), 'w') as f:
    json.dump(metrics, f)

combined_test_metrics = get_classification_metrics(
    test_df['label'], combined_model.predict(test_df.drop(columns=['label'])))
metrics = {
    'test_f1': combined_test_metrics['f1'],
    'test_acc': combined_test_metrics['accuracy'],
    'test_prec': combined_test_metrics['precision'],
    'test_rec': combined_test_metrics['recall']
}

with open(
        os.path.join(METRICS_DIR, 'combined_metrics.json'), 'w') as f:
    json.dump(metrics, f)
