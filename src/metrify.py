"""
    Stage to compile metrics from all historic models.
"""
import json
import os

from utils.misc import list_subdirectories, list_files


# Get list with all subdirectories in metrics/ (disregarding files). In this
# case, the subdirectories are the datasets.
METRICS_DIRS = list_subdirectories('metrics')

DATASETS = [d.split(os.sep)[-1] for d in METRICS_DIRS]

# Get list with all files in each subdirectory (for each dataset).
METRICS_FILES = {name: list_files(dir)
                 for dir, name in zip(METRICS_DIRS, DATASETS)}

metrics = {}
for dataset in DATASETS:
    metrics[dataset] = {}
    for file in METRICS_FILES[dataset]:
        with open(file, 'r') as f:
            file = file.split(os.sep)[-1].split('.')[0]
            metrics[dataset][file] = json.load(f)

with open(os.path.join('metrics', 'metrics.json'), 'w') as f:
    json.dump(metrics, f)
