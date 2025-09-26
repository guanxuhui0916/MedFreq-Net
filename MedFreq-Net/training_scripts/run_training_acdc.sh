#!/bin/sh

DATASET_PATH=../DATASET_Acdc

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_acdc
export medfreq_net_preprocessed="$DATASET_PATH"/medfreq_net_raw/medfreq_net_raw_data/Task01_ACDC
export medfreq_net_raw_data_base="$DATASET_PATH"/medfreq_net_raw

python ../medfreq_net/run/run_training.py 3d_fullres medfreq_net_trainer_acdc 1 0
