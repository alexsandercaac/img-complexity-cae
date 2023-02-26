"""
    The dataset is not split into train and test set. This script will split
    the dataset into train and test set. The split will be saved in the folder
    data/processed/train and data/processed/test. Class balance will be kept.
"""
from utils.dvc.params import get_params
import os
from rich import print, pretty
import shutil
import random
pretty.install()

# * Parameters
params = get_params()

SPLIT_RATIO = params['split_ratio']
SHUFFLE = params['shuffle']
SEED = params['seed']

# * Get file names from raw data for each class

file_names_ok = os.listdir('data/raw/casting_512x512/ok_front')
file_names_def = os.listdir('data/raw/casting_512x512/def_front')

print(f"Found {len(file_names_ok)} ok_front images (negative class).")
print(f"Found {len(file_names_def)} def_front images (positive class).")

# * Split the data into train and test set

if SHUFFLE:
    random.seed(SEED)
    random.shuffle(file_names_ok)
    random.shuffle(file_names_def)

train_split_index_ok = int(len(file_names_ok) * SPLIT_RATIO[0])
train_split_index_def = int(len(file_names_def) * SPLIT_RATIO[0])

val_split_index_ok = int(
    len(file_names_ok) * (SPLIT_RATIO[0] + SPLIT_RATIO[1])
)
val_split_index_def = int(
    len(file_names_def) * (SPLIT_RATIO[0] + SPLIT_RATIO[1])
)

train_ok = file_names_ok[:train_split_index_ok]
train_def = file_names_def[:train_split_index_def]

val_ok = file_names_ok[train_split_index_ok:val_split_index_ok]
val_def = file_names_def[train_split_index_def:val_split_index_def]

test_ok = file_names_ok[val_split_index_ok:]
test_def = file_names_def[val_split_index_def:]

print(f"Train set contains {len(train_ok)} ok_front images (negative class).")
print(
    f"Train set contains {len(train_def)} def_front images (positive class).")
print(
    f"Validation set contains {len(val_ok)} ok_front images (negative class).")
print(
    f"Validation set contains {len(val_def)} " +
    "def_front images (positive class).")
print(f"Test set contains {len(test_ok)} ok_front images (negative class).")
print(f"Test set contains {len(test_def)} def_front images (positive class).")

# * Save the split into the folder data/processed/train_test_split

if not os.path.exists('data/processed/train'):
    os.makedirs('data/processed/train')
    if not os.path.exists('data/processed/train/ok_front'):
        os.makedirs('data/processed/train/ok_front')
    if not os.path.exists('data/processed/train/def_front'):
        os.makedirs('data/processed/train/def_front')

if not os.path.exists('data/processed/val'):
    os.makedirs('data/processed/val')
    if not os.path.exists('data/processed/val/ok_front'):
        os.makedirs('data/processed/val/ok_front')
    if not os.path.exists('data/processed/val/def_front'):
        os.makedirs('data/processed/val/def_front')

if not os.path.exists('data/processed/test'):
    os.makedirs('data/processed/test')
    if not os.path.exists('data/processed/test/ok_front'):
        os.makedirs('data/processed/test/ok_front')
    if not os.path.exists('data/processed/test/def_front'):
        os.makedirs('data/processed/test/def_front')

for file_name in train_ok:
    shutil.copyfile(
        f"data/raw/casting_512x512/ok_front/{file_name}",
        f"data/processed/train/ok_front/{file_name}"
    )

for file_name in train_def:
    shutil.copyfile(
        f"data/raw/casting_512x512/def_front/{file_name}",
        f"data/processed/train/def_front/{file_name}"
    )

for file_name in val_ok:
    shutil.copyfile(
        f"data/raw/casting_512x512/ok_front/{file_name}",
        f"data/processed/val/ok_front/{file_name}"
    )

for file_name in val_def:
    shutil.copyfile(
        f"data/raw/casting_512x512/def_front/{file_name}",
        f"data/processed/val/def_front/{file_name}"
    )

for file_name in test_ok:
    shutil.copyfile(
        f"data/raw/casting_512x512/ok_front/{file_name}",
        f"data/processed/test/ok_front/{file_name}"
    )

for file_name in test_def:
    shutil.copyfile(
        f"data/raw/casting_512x512/def_front/{file_name}",
        f"data/processed/test/def_front/{file_name}"
    )
