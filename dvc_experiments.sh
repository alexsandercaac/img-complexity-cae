#!/usr/bin/env bash

dvc exp run --name 'CAE8' --set-param 'cae_hp_tuning.bottleneck_filters=8' --queue

dvc exp run --name 'CAE16' --set-param 'cae_hp_tuning.bottleneck_filters=16' --queue

dvc exp run --name 'CAE32' --set-param 'cae_hp_tuning.bottleneck_filters=32' --queue

dvc exp run --name 'CAE64' --set-param 'cae_hp_tuning.bottleneck_filters=64' --queue

dvc queue start
