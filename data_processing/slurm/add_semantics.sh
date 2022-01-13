#!/bin/bash

#SBATCH --job-name add_semantics
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/data_processing

python add_semantics.py -n $SLURM_ARRAY_TASK_COUNT -p $SLURM_ARRAY_TASK_ID
