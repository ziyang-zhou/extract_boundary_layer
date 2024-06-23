from antares import *
from functions import extract_BL_params
import vtk
import numpy as np
import os
import temporal
import pdb

vtu_path = temporal.vtu_path #path to read the vtu data from
vol_path = temporal.vol_path #path to write the h5 data to
step_per_chunk=temporal.step_per_chunk
total_timesteps=temporal.total_timesteps
starting_timestep = temporal.starting_timestep
time_step = 1
num_chunks = (total_timesteps-starting_timestep)//step_per_chunk
print('num chunks',num_chunks)
for n in range(num_chunks):
  #Read all instants from the PF data
  reader = Reader('bin_vtk')
  reader['filename'] = vtu_path + 'surface_mesh_frame_{:04d}_0.vtu'.format(step_per_chunk*n+starting_timestep)
  reader['shared'] = False
  b_vol = reader.read()
  for i in range(step_per_chunk*n+starting_timestep,step_per_chunk*(n+1)+starting_timestep,time_step):
    aux_reader = Reader('bin_vtk')
    aux_reader['filename'] = vtu_path + 'surface_mesh_frame_{:04d}_0.vtu'.format(i)
    base_aux = aux_reader.read()
    b_vol[0]['{:04d}'.format(i)] = Instant()
    b_vol[0]['{:04d}'.format(i)] = base_aux[0]['0000']
    del base_aux
    del aux_reader
    print("Instance {} read successfully.".format(i))
    print('max_x_velocity',np.max(b_vol[0]['{:04d}'.format(i)]['x_velocity']))
  
  del reader  
  writer = Writer('hdf_antares')
  writer['filename'] = vol_path + 'b_vol_{}_{}'.format(step_per_chunk*n+starting_timestep,step_per_chunk*(n+1)+starting_timestep)
  writer['base'] = b_vol
  writer.dump()
  print('completed chunk',n)
