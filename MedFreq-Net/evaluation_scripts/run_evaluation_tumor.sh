#!/bin/sh

DATASET_PATH=/mnt/d/downloads/MedFreq-Net/DATASET_Tumor

export PYTHONPATH=.././
export RESULTS_FOLDER=/mnt/d/downloads/MedFreq-Net/medfreq_net/evaluation/medfreq_net_tumor_checkpoint
export medfreq_net_preprocessed=/mnt/d/downloads/MedFreq-Net/DATASET_Tumor/medfreq_net_raw/medfreq_net_raw_data/Task03_tumor
export medfreq_net_raw_data_base=/mnt/d/downloads/MedFreq-Net/DATASET_Tumor/medfreq_net_raw

python /mnt/d/downloads/MedFreq-Net/medfreq_net/inference/predict_simple.py -i /mnt/d/downloads/MedFreq-Net/DATASET_Tumor/medfreq_net_raw/medfreq_net_raw_data/Task003_tumor/imagesTs -o /mnt/d/downloads/MedFreq-Net/medfreq_net/evaluation/medfreq_net_tumor_checkpoint/inferTs -m 3d_fullres  -t 3 -f 0 -chk model_final_checkpoint -tr medfreq_net_trainer_tumor


python /mnt/d/downloads/MedFreq-Net/medfreq_net/inference_tumor.py 0

