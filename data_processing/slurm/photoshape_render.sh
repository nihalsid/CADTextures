#!/bin/bash

#SBATCH --job-name render
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures

python data_processing/photoshape_render.py -i /cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs -m /cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres/ --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID