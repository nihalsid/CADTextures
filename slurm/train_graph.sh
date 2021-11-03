#!/bin/bash

#SBATCH --job-name base_filter64
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
##SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann,balrog,daidalos,gimli,himring,hithlum
#SBATCH --exclude=char,pegasus,tarsonis,daidalos,himring,hithlum
#SBATCH --partition=debug
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/

python -u trainer/train_graphnet.py dataset=graph_cube_textures_official experiment=cube_attn_sa_noblur_percept_lr_$lr_np_$num_pools  method=face conv=attention w_l1=35 w_content=0.1 w_style=1e4 wandb_main=True use_augmentations=False use_blur=False use_self_attn=True batch_size=4 nf=128 lr=$lr dataset.num_pools=$num_pools
