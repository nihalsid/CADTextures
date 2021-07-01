#!/bin/bash

#SBATCH --job-name texture_ifnet
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --gpus=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_ifnet.py sanity_steps=1 dataset=single_cube_textures dataset.preload=False experiment=$REXPERIMENT wandb_main=True val_check_interval=$RVALCHECK save_epoch=$RVALCHECK
