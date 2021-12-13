#!/bin/bash

#SBATCH --job-name supervise_2d
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,balrog,daidalos,gimli,himring,hithlum
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --partition=submit
#SBATCH --qos=normal

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/10.2

cd /rhome/ysiddiqui/CADTextures/

OMP_NUM_THREADS=4 python -u trainer/train_2d_supervision.py dataset=graph_cube_textures_official experiment=cube_attn_2d_supervision_base method=face conv=attention w_l1=35 w_content=0.1 w_style=1e4 wandb_main=True use_augmentations=False use_blur=False use_self_attn=True batch_size=4 nf=128 dataset.num_pools=3 max_epoch=500
