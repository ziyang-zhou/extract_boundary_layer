#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0-23:50

source /home/m/moreaust/zzhou/documents/env/bin/activate 
python script_clean_volume.py