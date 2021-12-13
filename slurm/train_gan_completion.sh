#!/bin/bash

#SBATCH --job-name texture_completion
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

python trainer/train_texture_completion.py sanity_steps=1 dataset=$RDATASET experiment=$REXPERIMENT wandb_main=True val_check_interval=$RVALCHECK save_epoch=$RVALCHECK
