#This script takes boundary layer time histories of u and v velocity components as input and outputs u_rms and v_rms

from antares import *
from functions import analysis
import vtk
import matplotlib.pyplot as plt
import numpy as np
import temporal
import os
import math
import h5py
import pandas as pd
import pdb

# ---------------------
# Defined functions
# ---------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def calc_rms(u):
	u_sq  = u**2
	u_mean_sq = np.mean(u_sq)
	u_rms = np.sqrt(u_mean_sq)
	return u_rms

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# ------------------
# Reading the files
# ------------------

settings = pd.read_csv("../setting.csv", index_col= 0)
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness

mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path
nb_points = temporal.nb_points #number of points across the boundary layer
var = 'U_n' #variable used for the cross correlation contour
timestep_size = temporal.timestep_size
xcoor0 = -0.019227 # x location of the integration axis
if_interpolate = True

# Set the total number of timesteps and the number of chunks
step_per_chunk = temporal.step_per_chunk
total_timesteps = temporal.total_timesteps
starting_timestep = temporal.starting_timestep
num_chunks = (total_timesteps - starting_timestep) // step_per_chunk


#Read the mesh
r=Reader('hdf_antares')
r['filename'] = mesh_read_path + 'interpolation_3d_grid.h5'
BL_line_geom=r.read()
print('shape of BL line geom',BL_line_geom[0][0]['x'].shape)

#For every timestep form a 2D matrix of velocities (wall normal , streamwise)
#Array dim : 1 is time, 2 is wall normal and 3 is chordwise
for j in range(num_chunks):
	#Read the boundary layer history at x_loc
	r = Reader('hdf_antares')
	r['filename'] = 'BL_line_prof/BL_line_prof_{}_{}.h5'.format(starting_timestep+j*step_per_chunk,starting_timestep+(j+1)*(step_per_chunk))
	BL_line_prof = r.read()
  
	if j == 0:
		#creation of streamwise distance array
		xcoor = BL_line_geom[0][0]['x'][:,0,0]
		ycoor = BL_line_geom[0][0]['y'][:,0,0] 
		ds = np.sqrt((xcoor[1:]-xcoor[:-1])**2 + (ycoor[1:]-ycoor[:-1])**2)
		sprof = np.zeros(ds.size+1,)
		sprof[1:] = np.cumsum(ds)
		scoor = sprof

		hcoor = BL_line_prof[0][0]['h'][0,:] # Read the wall normal distance

	for n,i in enumerate(BL_line_prof[0].keys()[1:]):
		for m in range(len(xcoor)):  # read all spatial locations in the current timestep
			profile_append = np.array(BL_line_prof[0][i][var][m])
			if (m==0):
				profile = profile_append #Create the data to be appended by concatenating the wall normal profiles for each x location
			elif (m==1):
				profile = np.concatenate((profile[:,np.newaxis], profile_append[:,np.newaxis]), axis=1)
			else :
				profile = np.concatenate((profile, profile_append[:,np.newaxis]), axis=1)

		data_append = profile
		if (n == 0) and (j == 0):
			data = data_append
		elif (n == 1) and (j == 0):
			data = np.concatenate((data[np.newaxis,:,:], data_append[np.newaxis,:,:]), axis=0)
		else:
			data = np.concatenate((data, data_append[np.newaxis,:,:]), axis=0)

	print('data shape is {}'.format(np.shape(data)))
	print('chunk {} read'.format(j))

if if_interpolate == True:
	for ki in range(0,np.shape(data)[2]-1):
		for t in range(0,np.shape(data)[0]-1):
			i_zero = np.where(abs(data[t,:,ki]) == 0)[0]
			for i in i_zero:
				if i+1 not in i_zero:
					data[t,i,ki] = (data[t,i-1,ki] + data[t,i+1,ki])/2
				else:
					#Obtain the next non-zero value for interpolation
					n=0
					while (data[t,i+n,ki]==0):
						n+=1
					data[t,i,ki] = (data[t,i-1,ki] + data[t,i+n,ki])/2

# ------------------------------
# Turbulence calculation
# ------------------------------

nbi = data.shape[0]
meandata = data.mean(axis=0,dtype=np.float64)

#Compute the arithmetic mean along the specified axis.
datafluc = data - np.tile(meandata,(nbi,1,1))
datarms = np.zeros((hcoor.shape[0],scoor.shape[0]))
#Define the x and y coord array
S,H =np.meshgrid(scoor,hcoor)

#Calculate the rms for each point in space
for ki in range(0,np.shape(datafluc)[2]):                                          #streamwise index_point
	for l in range(0,np.shape(datafluc)[1]):                                       #wall normal index_point
		u_prime = datafluc[:,l,ki]
		datarms[l,ki] = calc_rms(u_prime)

base_stat = Base()
base_stat['0'] = Zone()
base_stat['0']['0'] = Instant()
base_stat['0'].shared['s_coord'] = S
base_stat['0'].shared['h_coord'] = H
base_stat['0'].shared['u_rms'] = datarms

myw = Writer('hdf_antares')
myw['filename'] = 'BL_line_prof/{}_rms'.format(var)
myw['base'] = base_stat
myw.dump()