#!/bin/bash

#SBATCH --job-name texture_gan
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_texture_map_predictor.py  dataset.texture_map_size=128 sanity_steps=1 dataset=$RDATASET inputs=2d_only_partial_texture experiment=$REXPERIMENT model=$RMODEL wandb_main=True val_check_interval=$RVALCHECK save_epoch=$RVALCHECK
