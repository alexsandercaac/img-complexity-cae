"""
    This stage calculates the mean squared error of the autoencoder on the
    different datasets. The results are saved in a csv file.
"""
import os
import logging

import tensorflow as tf
from tqdm.rich import tqdm
from matplotlib import pyplot as plt

from utils.data.complexityaux import image_mse, load_imgs_gen

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


with open('data/processed/tabular/cae_mse.csv', 'w') as f:
    f.write('file,cae_mse,data_split,label\n')

DATA_SPLITS_AND_LABELS = [(split, label) for split in ['train', 'val', 'test']
                          for label in ['ok_front', 'def_front']]

model = tf.keras.models.load_model(
    filepath='models/best_cae.hdf5'
)

for split, label in DATA_SPLITS_AND_LABELS:
    files = [os.path.join(f'data/processed/{split}/{label}', f)
             for f in os.listdir(f'data/processed/{split}/{label}')]
    img_gen = load_imgs_gen(files, grayscale=True, scale=255)

    mses = []
    pbar = tqdm(total=len(files))
    for image in img_gen:
        pbar.update(1)
        pred = model.predict(image, verbose=0)
        mses.append(image_mse(image[0], pred[0]))

    pbar.close()
    if split != 'test':
        plt.hist(mses, bins=60)
        plt.title(f'CAE MSE for {split} {label}')
        plt.xlabel('MSE')
        plt.ylabel('Frequency')
        plt.savefig(f'visualisation/reconstruction/{split}_{label}_hist.png')
        plt.close()
    with open('data/processed/tabular/cae_mse.csv', 'a') as f:
        for file, mse in zip(files, mses):
            f.write(f'{file},{mse},{split},{label}\n')
