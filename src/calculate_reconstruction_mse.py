"""
    This stage calculates the mean squared error of the autoencoder on the
    different datasets. The results are saved in a csv file.
"""
import os
import logging

import tensorflow as tf
from tqdm.rich import tqdm
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from utils.data.complexityaux import image_mse, load_imgs_gen
from utils.dvc.params import get_params

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# * Parameters

params = get_params('all')

# Data parameters
DATASET = params['dataset']
GRAYSCALE = params['grayscale']
SCALE = params['scale']

# * Directories
# Initialize csv file that will store the reconstruction MSE for each image,
# along with the data split and label
OUTPUT_FILE = os.path.join(
    'data', 'processed', DATASET, 'tabular', 'cae_mse.csv')
FIG_DIR = os.path.join('visualisation', DATASET)
with open(OUTPUT_FILE, 'w') as f:
    f.write('file,cae_mse,data_split,label\n')

# Create a list of tuples containing the data split and label combinations
DATA_SPLITS_AND_LABELS = [(split, label) for split in ['train', 'val', 'test']
                          for label in ['positive', 'negative']]

model = tf.keras.models.load_model(
    filepath=os.path.join('models', DATASET, 'bin', 'trained_cae.hdf5')
)

fig, ax = plt.subplots()
cmap = plt.get_cmap('cool')
colours = {
    ('train', 'positive'): cmap(0.1),
    ('val', 'positive'): cmap(0.2),
    ('train', 'negative'): cmap(0.8),
    ('val', 'negative'): cmap(0.9),
}

for split, label in DATA_SPLITS_AND_LABELS:
    # Create a list of all the images of that split and label
    data_dir = os.path.join('data', 'processed', DATASET, split, label)
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    img_gen = load_imgs_gen(files, grayscale=GRAYSCALE, scale=SCALE)

    mses = []
    pbar = tqdm(total=len(files))
    for image in img_gen:
        pbar.update(1)
        pred = model.predict(image, verbose=0)
        mses.append(image_mse(image[0], pred[0]))

    pbar.close()
    if split != 'test':
        ax.hist(mses, bins=50, label=f'{split} {label}', alpha=0.7,
                color=colours[(split, label)])

    with open(OUTPUT_FILE, 'a') as f:
        for file, mse in zip(files, mses):
            f.write(f'{file},{mse},{split},{label}\n')

ax.set_xlabel('CAE MSE')
# Format the x-axis with powers of 10
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
ax.set_ylabel('Frequency')
ax.legend()

plt.savefig(os.path.join(FIG_DIR, 'caemse_hist.png'))
