"""
    Calculate reconstruction metrics for the CAE model in validation
    baseline datasets.

    Add average image complexity to each group for reference.
"""
import os
import json

import pandas as pd
import plotly.graph_objects as go

from utils.dvc.params import get_params
from utils.misc import transform_multilevel_header


# * Parameters

params = get_params('all')

DATASET = params['dataset']

cae_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'cae_mse.csv'),
    index_col=0
)
complexity_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'complexity.csv'),
    index_col=0)

complexity_df = complexity_df[['jpeg_mse', 'delentropy']]
cae_df = cae_df.join(complexity_df)

# Selecting validation and baseline datasets
mask = ((cae_df['data_split'] == 'val') | (cae_df['data_split'] == 'baseline'))
cae_df = cae_df[mask]
# Replace nan values with '-'
cae_df['label'] = cae_df['label'].fillna('-')

cae_df = cae_df.groupby(['data_split', 'label']).agg(['mean', 'std'])

# Transform cae_df index to columns
cae_df = cae_df.reset_index()

cae_df_single_level_header = transform_multilevel_header(cae_df)

# Write to csv in the metrics folder
cae_df_single_level_header.to_csv(
    os.path.join('metrics', DATASET, 'reconstruction_data.csv'),
    index=False
)

# DVC requires metrics to be stored in a json file, so we create a dict
cae_dict = {}

for _, row in cae_df.iterrows():
    cae_dict[row['data_split'][0] + ': ' + row['label'][0]] = {
        'cae_mse': {
            'mean': row['cae_mse']['mean'],
            'std': row['cae_mse']['std']
        },
        'jpeg_mse': {
            'mean': row['jpeg_mse']['mean'],
            'std': row['jpeg_mse']['std']
        },
        'delentropy': {
            'mean': row['delentropy']['mean'],
            'std': row['delentropy']['std']
        }
    }

with open(os.path.join(
        'metrics', DATASET, 'reconstruction_metrics.json'), 'w',
        encoding='utf-8') as f:
    json.dump(cae_dict, f)

# * Plotting

# Create bar graph with cae_df data
fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=cae_df['data_split'] + ': ' + cae_df['label'],
        y=cae_df['cae_mse']['mean'],
        error_y=dict(
            type='data',
            array=cae_df['cae_mse']['std'],
            visible=True
        ),
        name='CAE MSE'
    )
)

# Add jpeg_mse data with bars parallel to cae_mse bars
fig.add_trace(
    go.Bar(
        x=cae_df['data_split'] + ': ' + cae_df['label'],
        y=cae_df['jpeg_mse']['mean'],
        error_y=dict(
            type='data',
            array=cae_df['jpeg_mse']['std'],
            visible=True
        ),
        name='JPEG MSE'
    )
)

# Add delentropy data with bars parallel to cae_mse bars
fig.add_trace(
    go.Bar(
        x=cae_df['data_split'] + ': ' + cae_df['label'],
        y=cae_df['delentropy']['mean'],
        error_y=dict(
            type='data',
            array=cae_df['delentropy']['std'],
            visible=True
        ),
        name='Delentropy'
    )
)

fig.update_layout(
    title='CAE, JPEG MSE and Delentropy by dataset',
    xaxis_title='Dataset',
    yaxis_title='MSE',
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='rgba(255, 255, 255, 0.5)',
        borderwidth=1
    )
)

fig.write_html(
    os.path.join('visualisation', DATASET, 'reconstruction_metrics.html')
)
