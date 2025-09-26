#!/bin/sh

DATASET_PATH=/mnt/d/downloads/MedFreq-Net/DATASET_Acdc
CHECKPOINT_PATH=/mnt/d/downloads/MedFreq-Net/medfreq_net/evaluation/medfreq_net_acdc_checkpoint

export PYTHONPATH=.././
export RESULTS_FOLDER="$CHECKPOINT_PATH"
export medfreq_net_preprocessed="$DATASET_PATH"/medfreq_net_raw/medfreq_net_raw_data/Task01_ACDC
export medfreq_net_raw_data_base="$DATASET_PATH"/medfreq_net_raw

python /mnt/d/downloads/MedFreq-Net/medfreq_net/run/run_training.py 3d_fullres medfreq_net_trainer_acdc 1 0 -val


