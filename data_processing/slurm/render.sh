#!/bin/bash

#SBATCH --job-name blender-render
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --exclude=char,pegasus,seti,tarsonis

cd /rhome/ysiddiqui/CADTextures

python data_processing/create_renders_normals_df.py --blender /rhome/ysiddiqui/blender/blender --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID