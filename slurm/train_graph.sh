#!/bin/bash

#SBATCH --job-name graphnn
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --exclude=char,pegasus,tarsonis,balrog,daidalos,gimli,himring,hithlum
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python trainer/train_graphnet.py experiment=simple wandb_main=True
