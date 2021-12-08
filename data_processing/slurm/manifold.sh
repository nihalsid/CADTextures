#!/bin/bash

#SBATCH --job-name manifold
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --exclude=char,pegasus,seti,tarsonis

cd /rhome/ysiddiqui/CADTextures

python data_processing/manifold_mesh.py -i data/Photoshape-model/shapenet-chairs -o data/Photoshape-model/shapenet-chairs-manifold --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID