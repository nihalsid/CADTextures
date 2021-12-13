#!/bin/bash

#SBATCH --job-name quadriflow
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/data_processing
python photoshape.py -i /cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold/ --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID
