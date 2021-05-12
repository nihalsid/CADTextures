#!/bin/bash

#SBATCH --job-name naive_texturing
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_texture_map_predictor.py sanity_steps=1 dataset=$RDATASET inputs=$RINPUT experiment=$REXPERIMENT model=$RMODEL wandb_main=True val_check_interval=$RVALCHECK save_epoch=$RVALCHECK
