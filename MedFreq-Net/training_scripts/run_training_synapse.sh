#!/bin/sh

DATASET_PATH=../DATASET_Synapse

export PYTHONPATH=.././
export RESULTS_FOLDER=../output_synapse
export medfreq_net_preprocessed="$DATASET_PATH"/medfreq_net_raw/medfreq_net_raw_data/Task02_Synapse
export medfreq_net_raw_data_base="$DATASET_PATH"/medfreq_net_raw

python ../medfreq_net/run/run_training.py 3d_fullres medfreq_net_trainer_synapse 2 0
