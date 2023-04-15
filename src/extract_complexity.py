"""
    Stage to extract the complexity for each image in the dataset.
"""
from utils.data.jpegmse import calculate_jpeg_mse
from utils.data.complexityaux import load_imgs_gen
from utils.dvc.params import get_params
from utils.data.filemanipulation import create_dir

from tqdm.rich import tqdm
from matplotlib import pyplot as plt

import os

import matplotlib.ticker as ticker
import numpy as np


params = get_params()
stage_params = get_params()
all_params = get_params('all')

params = {**stage_params, **all_params}

DATASET = params['dataset']
GRAYSCLAE = params['grayscale']
SCALE = params['scale']

tabular_data_dir = os.path.join('data', 'processed', DATASET, 'tabular')
create_dir(tabular_data_dir)

file_name = 'complexity.csv'

with open(os.path.join(tabular_data_dir, file_name), 'w') as f:
    f.write('file,jpeg_mse,data_split,label\n')

DATA_SPLITS_AND_LABELS = [(split, label) for split in ['train', 'val', 'test']
                          for label in ['positive', 'negative']]

fig, ax = plt.subplots()
cmap = plt.get_cmap('cool')
colours = {
    ('train', 'positive'): cmap(0.1),
    ('val', 'positive'): cmap(0.2),
    ('train', 'negative'): cmap(0.8),
    ('val', 'negative'): cmap(0.9),
}

for split, label in DATA_SPLITS_AND_LABELS:
    data_dir = os.path.join('data', 'processed', DATASET, split, label)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    img_gen = load_imgs_gen(files, grayscale=GRAYSCLAE, scale=SCALE)

    mses = []
    pbar = tqdm(total=len(files))
    for image in img_gen:
        mses.append(calculate_jpeg_mse(image))
        pbar.update(1)
    pbar.close()
    if split != 'test':
        ax.hist(mses, bins=50, label=f'{split} {label}', alpha=0.7,
                color=colours[(split, label)])

    with open(os.path.join(tabular_data_dir, file_name), 'a') as f:
        for file, mse in zip(files, mses):
            f.write(f'{file},{mse},{split},{label}\n')

ax.set_xlabel('MSE')
# Format the x-axis with powers of 10
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
ax.set_ylabel('Frequency')
ax.legend()
fig_dir = os.path.join('visualisation', DATASET)
create_dir(fig_dir)

plt.savefig(os.path.join(fig_dir, 'complexity_hist.png'))
