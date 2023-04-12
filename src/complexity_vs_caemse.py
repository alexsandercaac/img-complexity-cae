"""
    In this stage, plots with image complexity data are generated.
"""
import pandas as pd
import plotly.graph_objects as go

complexity_df = pd.read_csv('data/processed/tabular/complexity.csv')
cae_df = pd.read_csv('data/processed/tabular/cae_mse.csv')

mask = ((complexity_df['data_split'] == 'train') |
        (complexity_df['data_split'] == 'val'))
complexity_df = complexity_df[mask].drop(columns=['data_split'])
mask = ((cae_df['data_split'] == 'train') |
        (cae_df['data_split'] == 'val'))
cae_df = cae_df[mask].drop(columns=['data_split', 'label'])

complexity_caemse_df = complexity_df.merge(cae_df, on='file')

complexity_caemse_df['label'] = complexity_caemse_df['label'].apply(
    lambda x: 1 if x == 'def_front' else 0)

fig = go.Figure()
# Plot complexity vs CAE MSE using labels as color
fig.add_trace(go.Scatter(
    x=complexity_caemse_df[complexity_caemse_df['label'] == 0]['jpeg_mse'],
    y=complexity_caemse_df[complexity_caemse_df['label'] == 0]['cae_mse'],
    mode='markers',
    name='Normal',
    marker=dict(
        color='blue',
        size=5
    )
))
fig.add_trace(go.Scatter(
    x=complexity_caemse_df[complexity_caemse_df['label'] == 1]['jpeg_mse'],
    y=complexity_caemse_df[complexity_caemse_df['label'] == 1]['cae_mse'],
    mode='markers',
    name='Defect',
    marker=dict(
        color='red',
        size=5
    )
))
fig.update_layout(
    title='Complexity vs CAE MSE',
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
rho = complexity_caemse_df['jpeg_mse'].corr(complexity_caemse_df['cae_mse'])

fig.add_annotation(
    x=complexity_caemse_df['jpeg_mse'].max(),
    y=complexity_caemse_df['cae_mse'].max(),
    text=f'Pearson correlation coefficient: {rho:.2f}',
    showarrow=False,
    font=dict(
        family='Courier New, monospace',
        size=18,
        color='#7f7f7f'
    )
)

fig.write_html('visualisation/casting_complexity_vs_caemse.html')
