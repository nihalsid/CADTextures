#!/bin/bash

#SBATCH --job-name texture_deform
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300gb
#SBATCH --gpus=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis
##SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_regression.py sanity_steps=1 dataset=single_cube_textures dataset.preload=True experiment=$REXPERIMENT wandb_main=True
