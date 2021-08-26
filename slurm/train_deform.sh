#!/bin/bash

#SBATCH --job-name texture_deform
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120gb
#SBATCH --gpus=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_multilevel_attn.py sanity_steps=1 dataset=single_cube_textures dataset.preload=False experiment=$REXPERIMENT wandb_main=True
