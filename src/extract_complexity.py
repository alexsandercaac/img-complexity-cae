"""
    Stage to extract the complexity for each image in the dataset.

    The complexity of a general purpose dataset is also
    extracted to be used as a baseline.
"""
from utils.data.jpegmse import jpeg_mse_complexity
from utils.data.complexityaux import load_imgs_gen
from utils.dvc.params import get_params
from utils.misc import create_dir

from tqdm.rich import tqdm
from matplotlib import pyplot as plt

import os

import matplotlib.ticker as ticker

# * Parameters


params = get_params('all')

# Data parameters
DATASET = params['dataset']
GRAYSCALE = params['grayscale']
SCALE = params['scale']

# * Directories
TABULAR_DATA_DIR = os.path.join('data', 'processed', DATASET, 'tabular')
create_dir(TABULAR_DATA_DIR)
FIG_DIR = os.path.join('visualisation', DATASET)
create_dir(FIG_DIR)

OUTPUT_FILE_NAME = 'complexity.csv'

# Initialise the output file
with open(os.path.join(TABULAR_DATA_DIR, OUTPUT_FILE_NAME), 'w') as f:
    f.write('file,jpeg_mse,data_split,label\n')
# Create a list of tuples containing the data split and label combinations
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

    img_gen = load_imgs_gen(files, grayscale=GRAYSCALE, scale=SCALE)

    mse_jpeg_complexity = []
    pbar = tqdm(total=len(files))
    for image in img_gen:
        mse_jpeg_complexity.append(jpeg_mse_complexity(image))
        pbar.update(1)
    pbar.close()
    if split != 'test':
        ax.hist(mse_jpeg_complexity, bins=50, label=f'{split} {label}',
                alpha=0.7, color=colours[(split, label)])

    with open(os.path.join(TABULAR_DATA_DIR, OUTPUT_FILE_NAME), 'a') as f:
        for file, mse in zip(files, mse_jpeg_complexity):
            f.write(f'{file},{mse},{split},{label}\n')

ax.set_xlabel('JPEG MSE')
# Format the x-axis with powers of 10
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
ax.set_ylabel('Frequency')
ax.legend()

plt.savefig(os.path.join(FIG_DIR, 'complexity_hist.png'))

# * Concat complexity of baseline images
# The baseline dataset comprises various images from open source datasets with
# no particular semantics. It is a general purpose dataset
data_dir = os.path.join(
    'data', 'raw', 'general', 'all')

files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

img_gen = load_imgs_gen(files, grayscale=GRAYSCALE, scale=SCALE)

mse_jpeg_complexity = []
pbar = tqdm(total=len(files))
for image in img_gen:
    mse_jpeg_complexity.append(calculate_jpeg_mse(image))
    pbar.update(1)
pbar.close()
with open(os.path.join(TABULAR_DATA_DIR, OUTPUT_FILE_NAME), 'a') as f:
    for file, mse in zip(files, mse_jpeg_complexity):
        f.write(f'{file},{mse},baseline,\n')
