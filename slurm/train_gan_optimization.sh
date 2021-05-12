#!/bin/bash

#SBATCH --job-name texture_optimization
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_patch_optimization.py sanity_steps=1 generator=$RGENERATOR dataset.texture_map_size=128 experiment=$REXPERIMENT wandb_main=True val_check_interval=1 save_epoch=1
