from antares import *
from functions import analysis
from scipy.signal import butter, lfilter
import vtk
import matplotlib.pyplot as plt
import numpy as np
import temporal
import os
import math
import pandas as pd
import pdb

# ---------------------
# Defined functions
# ---------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mov_avg(X,k):
    X_new = X
    for i in range(k//2,X_new.size-k//2):
        X_new[i] = sum(X[i-k//2:i+k//2])/k
    return X_new

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
var = 'U_n'
timestep_size = temporal.timestep_size

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

	for n,i in enumerate(BL_line_prof[0].keys()):
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


# ------------------------------
# Cross-correlation  calculation
# ------------------------------

nbi = data.shape[0]
meandata = data.mean(axis=0,dtype=np.float64)

#Compute the arithmetic mean along the specified axis.
datafluc = data - np.tile(meandata,(nbi,1,1))
pfluc = datafluc
Rxt_spectrum = np.zeros((hcoor.shape[0],scoor.shape[0]))

#Setting the fixed point
l0 = find_nearest_index(hcoor,0.1*delta_95) #wall normal coordinate of fixed point
ki0 = find_nearest_index(xcoor,-0.019227) #streamwise coordinate of fixed point

for ki in range(0,np.shape(pfluc)[2]):                                          #streamwise index_point
	for l in range(0,np.shape(pfluc)[1]):                                       #wall normal index_point
		p1 = pfluc[:,l,ki]
		p0 = pfluc[:,l0,ki0]
		time_cross_corr,cross_norm = analysis.get_pearson_corr(p1,p0,timestep_size)
		c = cross_norm[(len(cross_norm)-1)//2] #obtain value at zero time delay
		#argmax:Returns the indices of the maximum values along an axis.
		Rxt_spectrum[l,ki] = c

S,H =np.meshgrid((scoor-scoor[ki0])/delta_95,hcoor/delta_95)
fig,ax = plt.subplots(figsize=(5,8))

CS = ax.contourf(S, H, Rxt_spectrum,cmap='Greys')

ax.set_xlim([-0.2, 0.2])
ax.set_ylim([0, 0.4]) 
ax.set_xlabel(r'$X/delta^{95}$', fontsize=22)
ax.set_ylabel(r'$H/delta^{95}$', fontsize=22)

plt.savefig('velocity_corr_contour')

