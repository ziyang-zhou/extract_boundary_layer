from antares import *
from functions import analysis
import vtk
import matplotlib.pyplot as plt
import numpy as np
import temporal
import os
import math
import pandas as pd
import pdb

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def get_covariance(x, y):
    x_prime = x - np.mean(x)
    y_prime = y - np.mean(y)
    covariance = np.mean(np.multiply(x_prime,y_prime))
    return covariance

probe_list = [7,9,21,24]
x_loc_list = [-0.0809552,-0.0632463,-0.019227,-0.010625] #x locations of the probes where profiles are to be extracted (RMP 7,9,21,24)
mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path

#nb_point_chord = temporal.nb_point_chord #number of points along the chord
nb_points = temporal.nb_points #number of points across the boundary layer
total_timesteps=temporal.total_timesteps
starting_timestep = temporal.starting_timestep
step_per_chunk=temporal.step_per_chunk
num_chunks = (temporal.total_timesteps - starting_timestep)//temporal.step_per_chunk

#Read the mesh
r=Reader('hdf_antares')
r['filename'] = mesh_read_path + 'interpolation_3d_grid.h5'
BL_line_geom=r.read()
print('shape of BL line geom',BL_line_geom[0][0]['x'].shape)

#For every probe location, append the BL profile at each time step
for k,x_loc in enumerate(x_loc_list):
	tangential_velocity_magnitude_list = []
	normal_velocity_magnitude_list = []
	x_index = find_nearest_index(BL_line_geom[0][0]['x'][:, 0, 0],x_loc)

	for j in range(num_chunks): #read the current chunk (of timesteps)
		# read boundary layer data
		r = Reader('hdf_antares')
		r['filename'] = bl_read_path + 'BL_line_prof_{}_{}.h5'.format(starting_timestep + step_per_chunk * j, starting_timestep + (step_per_chunk * (j + 1)))
		BL_line_prof = r.read()
		print('shape of BL line prof',BL_line_prof[0][0]['U_t'].shape)
		ntimesteps = total_timesteps
		if j == 0:
			wall_distance = BL_line_prof[0][0]['h'][x_index]
		for n in range(starting_timestep + step_per_chunk * j, starting_timestep + (step_per_chunk * (j + 1))):  # read all timesteps in the current chunk
			tangential_velocity_magnitude_list.append(BL_line_prof[0]['{:04d}'.format(n)]['U_t'][x_index])
			normal_velocity_magnitude_list.append(BL_line_prof[0]['{:04d}'.format(n)]['U_n'][x_index])
	
	#Convert both sets of data in numpy arrays
	tangential_velocity_magnitude_list = np.array(tangential_velocity_magnitude_list)
	normal_velocity_magnitude_list = np.array(normal_velocity_magnitude_list)
	#Calculate the Reynolds stress and output it as a csv file
	uv = []
	for i,column in enumerate(tangential_velocity_magnitude_list[0,:]):	
		uv.append(get_covariance(tangential_velocity_magnitude_list[:,i],normal_velocity_magnitude_list[:,i]))
	
	#Display info on data shape and print data into a dataframe
	print('len y',len(wall_distance))
	print('uv',len(uv))
	data = {'y': wall_distance, 'uv': uv}
	df = pd.DataFrame(data)
	df.to_csv('BL_profile_extracted/probe_{}_uv_conformal.csv'.format(probe_list[k]), index=False)
	print('profile x={} completed'.format(x_loc))
	print('profile taken at x={}'.format(BL_line_geom[0][0]['x'][x_index,0,0]))
	
