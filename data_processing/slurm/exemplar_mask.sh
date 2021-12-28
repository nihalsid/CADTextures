#!/bin/bash

#SBATCH --job-name exemplar_mask
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
##SBATCH --exclude=char,tarsonis,pegasus
#SBATCH --partition=submit
#SBATCH --qos=normal

cd /rhome/ysiddiqui/CADTextures/data_processing

python concat_mask.py -i /cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars/ -o /cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars_mask -n $SLURM_ARRAY_TASK_COUNT -p $SLURM_ARRAY_TASK_ID
