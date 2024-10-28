#!/bin/bash

#SBATCH --job-name=extract_bl
#SBATCH --nodes=1
#SBATCH --tasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=0-11:50

source /project/m/moreaust/zzhou/pyenv/pyenv368_vtk/bin/activate
source /project/m/moreaust/zzhou/antares/v1.18.0/antares.env 
#python extract_boundary_layer.py
#python read_base_volume.py
python calc_correlation.py