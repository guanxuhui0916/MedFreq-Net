#!/bin/sh

DATASET_PATH=../DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_tumor
export medfreq_net_preprocessed="$DATASET_PATH"/medfreq_net_raw/medfreq_net_raw_data/Task03_tumor
export medfreq_net_raw_data_base="$DATASET_PATH"/medfreq_net_raw

python ../medfreq_net/run/run_training.py 3d_fullres medfreq_net_trainer_tumor 3 0

