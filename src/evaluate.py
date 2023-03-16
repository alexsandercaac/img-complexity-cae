"""
    In this stage, various classification metrics are extracted from the model.
"""
import pandas as pd
import json

from utils.evaluation.classification_metrics import get_classification_metrics


cae_df = pd.read_csv('data/processed/tabular/cae_mse.csv')
th = float(open('models/params/cae_threshold.txt', 'r').read())
cae_df['label'] = cae_df['label'].apply(
    lambda x: 1 if x == 'def_front' else 0)

mask = cae_df['data_split'] == 'train'
train_df = cae_df[mask].drop(columns=['data_split'])
mask = cae_df['data_split'] == 'val'
val_df = cae_df[mask].drop(columns=['data_split'])
mask = cae_df['data_split'] == 'test'
test_df = cae_df[mask].drop(columns=['data_split'])

train_predictions = train_df['cae_mse'].apply(lambda x: 1 if x > th else 0)
val_predictions = val_df['cae_mse'].apply(lambda x: 1 if x > th else 0)
test_predictions = test_df['cae_mse'].apply(lambda x: 1 if x > th else 0)

train_metrics = get_classification_metrics(
    train_df['label'], train_predictions)
train_f1 = train_metrics['f1']
train_acc = train_metrics['accuracy']
train_prec = train_metrics['precision']
train_rec = train_metrics['recall']

val_metrics = get_classification_metrics(
    val_df['label'], val_predictions)
val_f1 = val_metrics['f1']
val_acc = val_metrics['accuracy']
val_prec = val_metrics['precision']
val_rec = val_metrics['recall']

test_metrics = get_classification_metrics(
    test_df['label'], test_predictions)
test_f1 = test_metrics['f1']
test_acc = test_metrics['accuracy']
test_prec = test_metrics['precision']
test_rec = test_metrics['recall']

# Write metrics on json file
metrics = {
    'train_f1': train_f1,
    'train_acc': train_acc,
    'train_prec': train_prec,
    'train_rec': train_rec,
    'val_f1': val_f1,
    'val_acc': val_acc,
    'val_prec': val_prec,
    'val_rec': val_rec,
    'test_f1': test_f1,
    'test_acc': test_acc,
    'test_prec': test_prec,
    'test_rec': test_rec
}

with open('metrics/cae_metrics.json', 'w') as f:
    json.dump(metrics, f)