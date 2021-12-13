#!/bin/bash

#SBATCH --job-name texture_retrfuse
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160gb
#SBATCH --gpus=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
##SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_end2end_retrieval_fuse.py sanity_steps=1 dataset=single_cube_textures dataset.preload=True experiment=$REXPERIMENT wandb_main=True norm=$RNORM
