#!/bin/bash

#SBATCH --job-name graphdata
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=24G
#SBATCH --exclude=char,tarsonis,sorona,pegasus

cd /rhome/ysiddiqui/CADTextures

#python dataset/graph_mesh_dataset.py n_proc=$SLURM_ARRAY_TASK_COUNT proc=$SLURM_ARRAY_TASK_ID
python data_processing/process_mesh_for_conv.py -n $SLURM_ARRAY_TASK_COUNT -p $SLURM_ARRAY_TASK_ID