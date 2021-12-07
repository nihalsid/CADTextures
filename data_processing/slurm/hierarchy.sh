#!/bin/bash

#SBATCH --job-name hierarchy
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24gb
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yawar.siddiqui@tum.de
#SBATCH --exclude=char,pegasus,tarsonis,gondor,moria,seti,sorona,umoja,lothlann
#SBATCH --partition=debug
#SBATCH --qos=normal

source /usr/local/Modules/init/bash
source /usr/local/Modules/init/bash_completion
module load cuda/10.2

cd /rhome/ysiddiqui/CADTextures/data_processing/
# cd /rhome/ysiddiqui/sdf-gen/
# python process_hierarchy.py -i /cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold/ --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID
python select_hierarchy_level.py -i /cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold/ --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID
