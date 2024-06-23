from antares import *
from functions import analysis
import vtk
import matplotlib.pyplot as plt
import numpy as np
import temporal
import os
import math

mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path

total_timesteps=temporal.total_timesteps
starting_timestep = temporal.starting_timestep
step_per_chunk=temporal.step_per_chunk
num_chunks = (temporal.total_timesteps - starting_timestep)//temporal.step_per_chunk

cf_with_x_coord = analysis.cf_extraction(num_chunks,step_per_chunk,starting_timestep,total_timesteps,bl_read_path,mesh_read_path)
# Save cf_with_x_coord as a text file
np.savetxt('cf_with_x_coord.txt', cf_with_x_coord, delimiter='\t', fmt='%f')
