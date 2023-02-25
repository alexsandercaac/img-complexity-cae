"""
    Stage to extract the complexity for each image in the dataset.
"""
from utils.data.jpegmse import calculate_jpeg_mse
from utils.data.complexityaux import load_imgs_gen
import os
from tqdm.rich import tqdm
from matplotlib import pyplot as plt


with open('data/processed/tabular/complexity.csv', 'w') as f:
    f.write('file, jpeg_mse, data_split, label\n')

DATA_SPLITS_AND_LABELS = [(split, label) for split in ['train', 'test']
                          for label in ['ok_front', 'def_front']]

for split, label in DATA_SPLITS_AND_LABELS:
    files = [os.path.join(f'data/processed/{split}/{label}', f)
             for f in os.listdir(f'data/processed/{split}/{label}')]

    img_gen = load_imgs_gen(files, grayscale=True, scale=255)

    mses = []
    pbar = tqdm(total=len(files))
    for image in img_gen:
        mses.append(calculate_jpeg_mse(image))
        pbar.update(1)
    pbar.close()
    plt.hist(mses, bins=60)
    plt.savefig(f'visualisation/complexity/{split}_{label}.png')
    plt.title(f'JPEG MSE for {split} {label}')
    plt.xlabel('MSE')
    plt.ylabel('Frequency')
    plt.close()
    with open('data/processed/tabular/complexity.csv', 'a') as f:
        for file, mse in zip(files, mses):
            f.write(f'{file},{mse}, {split}, {label}\n')
