"""
    The dataset is not split into train and test set. This script will split
    the dataset into train and test set. The split will be saved in the folder
    data/processed/train and data/processed/test. Class balance will be kept.
"""
from rich import print, pretty

from utils.data.filemanipulation import create_dir, copy_files
from utils.data.misc import list_split
from utils.dvc.params import get_params

import os
import random

pretty.install()


# * Parameters
params = get_params()
stage_params = get_params()
all_params = get_params('all')

params = {**stage_params, **all_params}

SPLIT_RATIO = params['split_ratio']
SHUFFLE = params['shuffle']
SEED = params['seed']
DATASET = params['dataset']

# * Get file names from raw data for each class
positive_dir = os.path.join('data', 'raw', DATASET, 'positive')
negative_dir = os.path.join('data', 'raw', DATASET, 'negative')

file_names_positive = os.listdir(positive_dir)
file_names_negative = os.listdir(negative_dir)

print(f"Found {len(file_names_negative)} negative class images.")
print(f"Found {len(file_names_positive)} positive class images.")

# * Split the data into train and test set

if SHUFFLE:
    random.seed(SEED)
    random.shuffle(file_names_negative)
    random.shuffle(file_names_positive)

train_pos, val_pos, test_pos = list_split(
    file_names_positive, SPLIT_RATIO)
train_neg, val_neg, test_neg = list_split(
    file_names_negative, SPLIT_RATIO)

print(f"Train set contains {len(train_neg)} negative class images.")
print(
    f"Train set contains {len(train_pos)} positive class images.")
print(
    f"Validation set contains {len(val_neg)} negative class images.")
print(
    f"Validation set contains {len(val_pos)} " +
    "positive class images.")
print(f"Test set contains {len(test_neg)} negative class images.")
print(f"Test set contains {len(test_pos)} positive class images.")

# * Save the split into the folders at data/processed/$DATASET

train_dir = os.path.join('data', 'processed', DATASET, 'train')
subdirs = ['negative', 'positive']

create_dir(train_dir, subdirs)
copy_files(train_neg, negative_dir, os.path.join(train_dir, 'negative'))
copy_files(train_pos, positive_dir, os.path.join(train_dir, 'positive'))

val_dir = os.path.join('data', 'processed', DATASET, 'val')

create_dir(val_dir, subdirs)
copy_files(val_neg, negative_dir, os.path.join(val_dir, 'negative'))
copy_files(val_pos, positive_dir, os.path.join(val_dir, 'positive'))

test_dir = os.path.join('data', 'processed', DATASET, 'test')

create_dir(test_dir, subdirs)
copy_files(test_neg, negative_dir, os.path.join(test_dir, 'negative'))
copy_files(test_pos, positive_dir, os.path.join(test_dir, 'positive'))
