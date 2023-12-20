"""
    In this stage, various classification metrics are extracted from the model.
"""
import os
import json

import pandas as pd
import joblib
import sklearn

from utils.evaluation.classification_metrics import get_classification_metrics
from utils.dvc.params import get_params
from utils.misc import create_dir

params = get_params('all')

DATASET = params['dataset']
MODEL_DIR = os.path.join('models', DATASET, 'bin')
METRICS_DIR = os.path.join('metrics', DATASET)
VIS_DIR = os.path.join('visualisation', DATASET)
create_dir(METRICS_DIR)

cae_df = pd.read_csv(
    os.path.join(
        'data', 'processed', DATASET, 'tabular', 'complexity_caemse.csv'),
    index_col=0
)


mask = cae_df['data_split'] == 'test'
test_df = cae_df[mask].drop(columns=['data_split'])

# Load models
autoencoder_model = joblib.load(
    os.path.join(MODEL_DIR, 'logistic_regression_autoencoder.joblib'))
complexity_model = joblib.load(
    os.path.join(MODEL_DIR, 'logistic_regression_complexity.joblib'))
combined_model = joblib.load(
    os.path.join(MODEL_DIR, 'logistic_regression_combined.joblib'))


autoencoder_test_metrics = get_classification_metrics(
    test_df['label'],
    autoencoder_model.predict(test_df.drop(
        columns=['label', 'jpeg_mse', 'delentropy'])))

# Write metrics on json file
metrics = {
    'test_f1': autoencoder_test_metrics['f1'],
    'test_acc': autoencoder_test_metrics['accuracy'],
    'test_prec': autoencoder_test_metrics['precision'],
    'test_rec': autoencoder_test_metrics['recall']
}

with open(
        os.path.join(METRICS_DIR, 'autoencoder_metrics.json'), 'w') as f:
    json.dump(metrics, f)

cm_display = sklearn.metrics.ConfusionMatrixDisplay(
    confusion_matrix=autoencoder_test_metrics['conf_matrix'],
    display_labels=['Normal', 'Anomalous'])

cm_display.plot().figure_.savefig(
    os.path.join(VIS_DIR, 'autoencoder_confusion_matrix.png'))

complexity_test_metrics = get_classification_metrics(
    test_df['label'],
    complexity_model.predict(test_df.drop(columns=['label', 'cae_mse'])))
metrics = {
    'test_f1': complexity_test_metrics['f1'],
    'test_acc': complexity_test_metrics['accuracy'],
    'test_prec': complexity_test_metrics['precision'],
    'test_rec': complexity_test_metrics['recall']
}

with open(
        os.path.join(METRICS_DIR, 'complexity_metrics.json'), 'w') as f:
    json.dump(metrics, f)

cm_display = sklearn.metrics.ConfusionMatrixDisplay(
    confusion_matrix=complexity_test_metrics['conf_matrix'],
    display_labels=['Normal', 'Anomalous'])
cm_display.plot().figure_.savefig(
    os.path.join(VIS_DIR, 'complexity_confusion_matrix.png'))

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

cm_display = sklearn.metrics.ConfusionMatrixDisplay(
    confusion_matrix=combined_test_metrics['conf_matrix'],
    display_labels=['Normal', 'Anomalous'])
cm_display.plot().figure_.savefig(
    os.path.join(VIS_DIR, 'combined_confusion_matrix.png'))
