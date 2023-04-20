"""
    In this stage, the reconstruction error of the CAE, corrected by image
    complexity is used as an outlier score and prediction is made based on a
    threshold defined with the validation data.
"""
import os

import pandas as pd
from sklearn.metrics import f1_score
import plotly.graph_objects as go

from utils.models.threshold_search import bayesian_search_th
from utils.misc import create_dir
from utils.dvc.params import get_params


pd.options.plotting.backend = "plotly"

params = get_params()

DATASET = params['dataset']
TH_DIR = os.path.join('models', DATASET, 'params')
create_dir(TH_DIR)

# Algorithm parameters
SCORE_FUNC = params['score_func']
N_ITER = params['n_iter']
BALANCED = params['balanced']

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

mask = cae_df['data_split'] == 'val'
val_df = cae_df[mask].drop(columns=['data_split'])


if SCORE_FUNC == 'accuracy':
    score_func = None
elif SCORE_FUNC == 'f1':
    score_func = f1_score
else:
    raise ValueError(
        f"Invalid score function {SCORE_FUNC} specified.")

val_df['corrected_mse'] = val_df['cae_mse'] / val_df['jpeg_mse']

search_results = bayesian_search_th(
    val_df['corrected_mse'].values, val_df['label'].values,
    val_df['corrected_mse'].min(), val_df['corrected_mse'].max(),
    n_iter=N_ITER, score_func=score_func,
    balanced=BALANCED, verbose=3
)

# Write best threshold to file in models/params
with open(os.path.join(TH_DIR, 'corrected_mse_threshold.txt'), 'w') as f:
    f.write(str(search_results['threshold']))

# Plot histogram of corrected_mse with different colors according to label
fig = val_df['corrected_mse'].hist(
    by=val_df['label'], color=val_df['label'].apply(
        lambda x: 'positive' if x == 1 else 'negative'), opacity=0.6)

fig.add_trace(
    go.Scatter(
        x=[search_results['threshold'], search_results['threshold']],
        y=[0, 20],
        mode="lines",
        name="Threshold",
        line=dict(color='red', width=3, dash='dash'),
    )
)

fig.update_layout(
    title_text="Histogram of Corrected CAE-MSE",
    xaxis_title_text="Corrected CAE-MSE",
    yaxis_title_text="Count",
    legend_title_text="Legend",
    font=dict(
        family="Courier New, monospace",
        size=18,
    )
)

fig.write_html(
    os.path.join('visualisation', DATASET, 'corrected_mse.html'))
