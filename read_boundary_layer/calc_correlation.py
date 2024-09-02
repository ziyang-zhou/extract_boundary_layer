# This script takes in the boundary layer profile output outputs the integral length scale of wall normal velocity fluctuation 
from antares import *
from functions import analysis
from scipy.signal import butter, lfilter, savgol_filter
from scipy import interpolate, integrate
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
# ------------------
# Reading the files
# ------------------

settings = pd.read_csv("../setting.csv", index_col= 0)
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness

mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path
nb_points = temporal.nb_points #number of points across the boundary layer
var = temporal.var
timestep_size = temporal.timestep_size
fs = 1/timestep_size

# Set the total number of timesteps and the number of chunks
step_per_chunk = temporal.step_per_chunk
total_timesteps = temporal.total_timesteps
starting_timestep = temporal.starting_timestep
num_chunks = (total_timesteps - starting_timestep) // step_per_chunk

if_interpolate = temporal.if_interpolate # Set as true if interpolation is needed to remove zeros in the contour
if_integrate = temporal.if_integrate # Set as true if the integral is to be calculated
troubleshoot = temporal.troubleshoot # Set as true if velocity contours are to be plotted
integration_axis = temporal.integration_axis
xcoor0 = temporal.xcoor0 # x location of the integration axis

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
	r['filename'] = bl_read_path + 'BL_line_prof_{}_{}.h5'.format(starting_timestep+j*step_per_chunk,starting_timestep+(j+1)*(step_per_chunk))
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


# ------------------------------
# Cross-correlation  calculation
# ------------------------------

nbi = data.shape[0]
meandata = data.mean(axis=0,dtype=np.float64)

#Setting the fixed point
l0 = analysis.find_nearest(hcoor,0.1*delta_95) #wall normal coordinate of fixed point
ki0 = analysis.find_nearest(xcoor,xcoor0) #streamwise coordinate of fixed point
h_mask_delta_95 = (hcoor < delta_95) #Mask to scope out the boundary layer

#Compute the arithmetic mean along the specified axis.
pfluc = data - np.tile(meandata,(nbi,1,1))
Rxt_spectrum = np.zeros((hcoor.shape[0],scoor.shape[0]))

#Smooth interpolated error
if if_interpolate == True:
	for ki in range(0,np.shape(pfluc)[2]-1):
		for t in range(0,np.shape(pfluc)[0]-1):
			i_zero = np.where(abs(pfluc[t,:,ki]) == 0)[0]
			for i in i_zero:
				if i+1 not in i_zero:
					pfluc[t,i,ki] = (pfluc[t,i-1,ki] + pfluc[t,i+1,ki])/2
				else:
					#Obtain the next non-zero value for interpolation
					n=0
					while (pfluc[t,i+n,ki]==0):
						n+=1
					pfluc[t,i,ki] = (pfluc[t,i-1,ki] + pfluc[t,i+n,ki])/2

#Calculate the cross correlation contour
for ki in range(0,np.shape(pfluc)[2]-1):                                          #streamwise index_point
	for l in range(0,np.shape(pfluc)[1]-1):                                       #wall normal index_point
		p1 = pfluc[:,l,ki]
		p0 = pfluc[:,l0,ki0]
		c = analysis.get_velocity_corr(p0,p1,hcoor[l0],hcoor[l])
		Rxt_spectrum[l,ki] = c

print('shape of the matrix is',pfluc.shape)
print('format is (time,wall normal,streamwise)')

#Define the limits of the plot
S,H =np.meshgrid((scoor-scoor[ki0])/delta_95,hcoor/delta_95)
fig,ax = plt.subplots(figsize=(5,8))

# Contour plot with black lines
levels = np.linspace(0.1, 1.0, 9)
CS = ax.contour(S, H, Rxt_spectrum, levels=levels, colors='black')
plt.clabel(CS, fmt='%1.1f', inline=True, fontsize=10)
ax.set_xlim([-0.15, 0.15])
ax.set_ylim([0, 0.4]) 
ax.set_xlabel(r'$X/delta^{95}$', fontsize=22)
ax.set_ylabel(r'$H/delta^{95}$', fontsize=22)
interval = 20
levels = np.linspace(-25, 25, 51)
plt.savefig(temporal.project_path + 'velocity_corr_contour')

#Calculate integral length scale 22+
h_start = 0.0*delta_95 #start location of the fixed point
h_end = 1.0*delta_95 #end location of the fixed point
h_mask_plot_range = ((hcoor < h_end) & (hcoor > h_start))
h_mask_integrate_range = (hcoor > h_start)

if if_integrate == True:
	L_22, scale = analysis.get_length_scale(pfluc,scoor,hcoor,scoor[ki0],h_start,scoor[ki0],h_end,axis=integration_axis)
	L_22_df = pd.DataFrame({'wall distance': scale, 'L22+':L_22})
	L_22_df.to_csv(temporal.project_path + 'L22+_{}'.format(integration_axis),index=False)

#Plot the contour

if troubleshoot == True:
	levels = np.linspace(-2.5, 2.5, 21)  # 20 levels from 0 to 10
	cmap = 'rainbow'
	timestep_interval = 100
	for t in range(0,np.shape(pfluc)[0],timestep_interval):
		print('creating contour of timestep {}'.format(t))
		fig,ax = plt.subplots(figsize=(5,8))
		
		CS = ax.contourf(S, H, pfluc[t,:,:], levels=levels, cmap=cmap)
		ax.set_xlim([-0.2, 0.2])
		ax.set_ylim([0, 0.5]) 
		ax.set_xlabel(r'$X/delta^{95}$', fontsize=22)
		ax.set_ylabel(r'$H/delta^{95}$', fontsize=22)
		# Add a colorbar
		cbar = plt.colorbar(CS, ax=ax)
		cbar.set_label('Velocity', fontsize=18)
		plt.savefig(temporal.project_path + 'velocity_corr_contour timestep {}'.format(t))
		plt.close()

	for t in range(0,np.shape(pfluc)[0],timestep_interval):
		fig,ax = plt.subplots(figsize=(5,8))	
		x_index = analysis.find_nearest(scoor,0.1*delta_95)
		plt.plot(hcoor/delta_95,pfluc[t,:,x_index])
		plt.ylim((-2.5,2.5))
		plt.savefig(temporal.project_path + 'fluc plot {}'.format(t))
		plt.close()
else:
	None
