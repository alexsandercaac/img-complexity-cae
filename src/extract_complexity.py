"""
    Stage to extract the complexity for each image in the dataset.

    The complexity of a general purpose dataset is also
    extracted to be used as a baseline.
"""
import os

from matplotlib import ticker, pyplot as plt
from tqdm.rich import tqdm

from utils.data.jpegmse import jpeg_mse_complexity
from utils.data.entropy import delentropy_complexity
from utils.data.complexityaux import load_imgs_gen
from utils.dvc.params import get_params
from utils.misc import create_dir


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
    f.write('file,jpeg_mse,delentropy,data_split,label\n')
# Create a list of tuples containing the data split and label combinations
DATA_SPLITS_AND_LABELS = [(split, label) for split in ['train', 'val', 'test']
                          for label in ['positive', 'negative']]

fig, (ax_jpeg, ax_delentropy) = plt.subplots(2)

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
    delentropy_complexity_list = []
    pbar = tqdm(total=len(files))
    for image in img_gen:
        mse_jpeg_complexity.append(jpeg_mse_complexity(image))
        delentropy_complexity_list.append(delentropy_complexity(image))
        pbar.update(1)
    pbar.close()
    if split != 'test':
        ax_jpeg.hist(mse_jpeg_complexity, bins=50, label=f'{split} {label}',
                     alpha=0.7, color=colours[(split, label)])
        ax_delentropy.hist(
            delentropy_complexity_list, bins=50, label=f'{split} {label}',
            alpha=0.7, color=colours[(split, label)])

    with open(os.path.join(TABULAR_DATA_DIR, OUTPUT_FILE_NAME), 'a') as f:
        files_mses_entropies = zip(files, mse_jpeg_complexity,
                                   delentropy_complexity_list)
        for file, mse, delentropy in files_mses_entropies:
            f.write(f'{file},{mse},{delentropy},{split},{label}\n')

ax_jpeg.set_xlabel('JPEG MSE')
# Format the x-axis with powers of 10
ax_jpeg.xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x:.1e}"))
ax_jpeg.set_ylabel('Frequency')
ax_jpeg.legend()

ax_delentropy.set_xlabel('Delentropy')
ax_delentropy.set_ylabel('Frequency')
fig.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'complexity_hist.png'))

# * Concat complexity of baseline images
# The baseline dataset comprises various images from open source datasets with
# no particular semantics. It is a general purpose dataset
data_dir = os.path.join(
    'data', 'raw', 'general', 'all')

files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

img_gen = load_imgs_gen(files, grayscale=GRAYSCALE, scale=SCALE)

mse_jpeg_complexity = []
delentropy_complexity_list = []
pbar = tqdm(total=len(files))
for image in img_gen:
    mse_jpeg_complexity.append(jpeg_mse_complexity(image))
    delentropy_complexity_list.append(delentropy_complexity(image))
    pbar.update(1)
pbar.close()
with open(os.path.join(TABULAR_DATA_DIR, OUTPUT_FILE_NAME), 'a') as f:
    files_mses_entropies = zip(files, mse_jpeg_complexity,
                               delentropy_complexity_list)
    for file, mse, delentropy in files_mses_entropies:
        f.write(f'{file},{mse},{delentropy},baseline,\n')
