"""
    In this stage, plots with image complexity data are generated.
"""
import os

import pandas as pd
import plotly.graph_objects as go

from utils.dvc.params import get_params


params = get_params('all')

DATASET = params['dataset']
FIG_DIR = os.path.join('visualisation', DATASET)

complexity_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'complexity.csv'),
    dtype={'file': str, 'label': str, 'data_split': str,
           'jpeg_mse': float, 'delentropy': float})

cae_df = pd.read_csv(
    os.path.join('data', 'processed', DATASET, 'tabular', 'cae_mse.csv'),
    dtype={'file': str, 'label': str, 'data_split': str,
           'cae_mse': float})

complexity_df = complexity_df.drop(columns=['data_split', 'label'])

complexity_caemse_df = complexity_df.merge(cae_df, on='file')

complexity_caemse_df['label'] = complexity_caemse_df['label'].apply(
    lambda x: 1 if x == 'positive' else 0)

complexity_caemse_df.to_csv(
    os.path.join(
        'data', 'processed', DATASET, 'tabular', 'complexity_caemse.csv'),
    index=False
)

print('Complexity vs CAE MSE data saved to tabular folder.')

# *** Plot complexity vs CAE MSE ***

drop_test_mask = ((complexity_caemse_df['data_split'] == 'val') | (
    complexity_caemse_df['data_split'] == 'train'))

fig_jpeg = go.Figure()

normal_mask = (complexity_caemse_df['label'] == 0) & drop_test_mask
fig_jpeg.add_trace(go.Scatter(
    x=complexity_caemse_df[normal_mask]['jpeg_mse'],
    y=complexity_caemse_df[normal_mask]['cae_mse'],
    mode='markers',
    name='Normal',
    marker=dict(
        color='blue',
        size=5
    )
))

defect_mask = (complexity_caemse_df['label'] == 1) & drop_test_mask
fig_jpeg.add_trace(go.Scatter(
    x=complexity_caemse_df[defect_mask]['jpeg_mse'],
    y=complexity_caemse_df[defect_mask]['cae_mse'],
    mode='markers',
    name='Defect',
    marker=dict(
        color='red',
        size=5
    )
))

baseline_mask = complexity_caemse_df['data_split'] == 'baseline'
fig_jpeg.add_trace(go.Scatter(
    x=complexity_caemse_df[baseline_mask]['jpeg_mse'],
    y=complexity_caemse_df[baseline_mask]['cae_mse'],
    mode='markers',
    name='Baseline',
    marker=dict(
        color='green',
        size=5
    )
))

fig_jpeg.update_layout(
    title='JPEG Complexity vs CAE MSE',
    xaxis_title='Complexity',
    yaxis_title='CAE MSE',
    width=800,
    height=600,
    showlegend=True,
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)
rho = complexity_caemse_df[drop_test_mask]['jpeg_mse'].corr(
    complexity_caemse_df[drop_test_mask]['cae_mse'])

fig_jpeg.add_annotation(
    x=complexity_caemse_df[drop_test_mask]['jpeg_mse'].max(),
    y=complexity_caemse_df[drop_test_mask]['cae_mse'].max(),
    text=f'Pearson correlation coefficient: {rho:.2f}',
    showarrow=False,
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)

fig_delentropy = go.Figure()

fig_delentropy.add_trace(go.Scatter(
    x=complexity_caemse_df[normal_mask]['delentropy'],
    y=complexity_caemse_df[normal_mask]['cae_mse'],
    mode='markers',
    name='Normal',
    marker=dict(
        color='blue',
        size=5
    )
))
fig_delentropy.add_trace(go.Scatter(
    x=complexity_caemse_df[defect_mask]['delentropy'],
    y=complexity_caemse_df[defect_mask]['cae_mse'],
    mode='markers',
    name='Defect',
    marker=dict(
        color='red',
        size=5
    )
))
fig_delentropy.add_trace(go.Scatter(
    x=complexity_caemse_df[baseline_mask]['delentropy'],
    y=complexity_caemse_df[baseline_mask]['cae_mse'],
    mode='markers',
    name='Baseline',
    marker=dict(
        color='green',
        size=5
    )
))

fig_delentropy.update_layout(
    title='Delentropy vs CAE MSE',
    xaxis_title='Complexity',
    yaxis_title='CAE MSE',
    width=800,
    height=600,
    showlegend=True,
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)
rho = complexity_caemse_df[drop_test_mask]['delentropy'].corr(
    complexity_caemse_df[drop_test_mask]['cae_mse'])

fig_delentropy.add_annotation(
    x=complexity_caemse_df['delentropy'].max(),
    y=complexity_caemse_df['cae_mse'].max(),
    text=f'Pearson correlation coefficient: {rho:.2f}',
    showarrow=False,
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)

with open(os.path.join(FIG_DIR, 'complexity_vs_caemse.html'), 'w') as f:
    f.write(fig_jpeg.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write(fig_delentropy.to_html(full_html=False, include_plotlyjs='cdn'))
