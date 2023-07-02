"""
    The raw datasets are not inherently divided into train, validation and test
    sets. As such, this script will split the dataset, saving the outputs in
    the data/processed/$DATASET folders. Class balance will be kept.
"""
from rich import print, pretty

from utils.misc import list_split, create_dir, copy_files
from utils.dvc.params import get_params

import os
import random

pretty.install()


# * Parameters
params = get_params()

SPLIT_RATIO = params['split_ratio']
SHUFFLE = params['shuffle']
SEED = params['seed']
DATASET = params['dataset']

# * Directories
POSITIVE_DIR = os.path.join('data', 'raw', DATASET, 'positive')
NEGATIVE_DIR = os.path.join('data', 'raw', DATASET, 'negative')
TRAIN_DIR = os.path.join('data', 'processed', DATASET, 'train')
VAL_DIR = os.path.join('data', 'processed', DATASET, 'val')
TEST_DIR = os.path.join('data', 'processed', DATASET, 'test')
SUBDIRS = ['negative', 'positive']

file_names_positive = os.listdir(POSITIVE_DIR)
file_names_negative = os.listdir(NEGATIVE_DIR)

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

create_dir(TRAIN_DIR, SUBDIRS)
copy_files(train_neg, NEGATIVE_DIR, os.path.join(TRAIN_DIR, 'negative'))
copy_files(train_pos, POSITIVE_DIR, os.path.join(TRAIN_DIR, 'positive'))

create_dir(VAL_DIR, SUBDIRS)
copy_files(val_neg, NEGATIVE_DIR, os.path.join(VAL_DIR, 'negative'))
copy_files(val_pos, POSITIVE_DIR, os.path.join(VAL_DIR, 'positive'))

create_dir(TEST_DIR, SUBDIRS)
copy_files(test_neg, NEGATIVE_DIR, os.path.join(TEST_DIR, 'negative'))
copy_files(test_pos, POSITIVE_DIR, os.path.join(TEST_DIR, 'positive'))
