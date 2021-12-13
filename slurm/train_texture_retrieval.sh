#!/bin/bash

#SBATCH --job-name texture_retrievaal
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_patch_retrieval.py experiment=$REXPERIMENT wandb_main=True dataset=$RDATASET temperature=$RTEMPERATURE val_check_interval=$RVALCHECK save_epoch=$RVALCHECK
