#!/bin/bash

#SBATCH --job-name ifnet_data_2d
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --exclude=char,pegasus,seti,tarsonis

cd /rhome/ysiddiqui/CADTextures

python data_processing/create_ifnet_data.py --dataset SingleShape/CubeTextures --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID