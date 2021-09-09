#!/bin/bash

#SBATCH --job-name subdivision
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --exclude=char,pegasus,seti,tarsonis

cd /rhome/ysiddiqui/CADTextures

python data_processing/subdivide_mesh.py -i /cluster_HDD/pegasus/yawar/ShapeNetCore.v2/03001627/ -o /cluster/gimli/ysiddiqui/CADTextures/ShapeNetV2/VertexColor_0_15/03001627 --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID