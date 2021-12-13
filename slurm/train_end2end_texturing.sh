#!/bin/bash

#SBATCH --job-name texture_e2e
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80gb
#SBATCH --gpus=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --exclude=char,pegasus,tarsonis
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_end2end_retrieval.py sanity_steps=1 dataset=single_cube_textures dataset.preload=True dictionary.patch_size=$RPATCHSIZE experiment=$REXPERIMENT wandb_main=True val_check_interval=$RVALCHECK save_epoch=$RVALCHECK warmup_epochs_constrastive=$RWARMUP
