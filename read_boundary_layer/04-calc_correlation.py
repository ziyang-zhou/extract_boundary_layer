# This script takes in the boundary layer profile output outputs the integral length scale of wall normal velocity fluctuation 
from antares import *
from functions import analysis
from functions import extract_BL_params
from scipy.signal import butter, lfilter, savgol_filter
from scipy import interpolate, integrate
from scipy.optimize import curve_fit
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

# Define the exponential function
def model(x, L):
    return np.exp(-x / L)
# ------------------
# Reading the files
# ------------------

settings = pd.read_csv("../setting.csv", index_col= 0)
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness
le_cut = eval(settings.at["le_cut", settings.columns[0]])
te_cut = eval(settings.at["te_cut", settings.columns[0]])

project_path = temporal.project_path
mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path #Place to read BL_prof base
bl_save_path = temporal.project_path + 'boundary_layer_profile/' #Place to save boundary layer dataframe
result_save_path = project_path + 'result/'
os.makedirs(result_save_path, exist_ok=True)

probe_number = temporal.probe_number
nb_points = temporal.nb_points #number of points across the boundary layer
var = temporal.var
timestep_size = temporal.timestep_size
fs = 1/timestep_size
timestep_interval = temporal.timestep_interval
case_name = 'L08'

# Set the total number of timesteps and the number of chunks
step_per_chunk = temporal.step_per_chunk
total_timesteps = temporal.total_timesteps
starting_timestep = temporal.starting_timestep
num_chunks = (total_timesteps - starting_timestep) // step_per_chunk

if_interpolate = temporal.if_interpolate # Set as true if interpolation is needed to remove zeros in the contour
if_integrate_axis = temporal.if_integrate_axis # Set as true if the integral length scale along axis is to be calculated
if_integrate_field = temporal.if_integrate_field # Set as true if the integral length scale field is to be calculated
if_read_boundary_velocity = temporal.if_read_boundary_velocity # Set as true if the boundary layer mean is to be computed
troubleshoot = temporal.troubleshoot # Set as true if velocity contours are to be plotted
xcoor0 = temporal.xcoor0 # x location of the integration axis

# Create required directory
probe_save_path = result_save_path + 'probe_{}/'.format(probe_number)
os.makedirs(probe_save_path, exist_ok=True)

#Read the mesh
r=Reader('hdf_antares')
r['filename'] = mesh_read_path + 'interpolation_3d_grid.h5'
BL_line_geom=r.read()
print('shape of BL line geom',BL_line_geom[0][0]['x'].shape)

#For every timestep form a 2D matrix of velocities (wall normal , streamwise)
#Array dim : 1 is time, 2 is wall normal and 3 is chordwise
pfluc_path = bl_read_path + 'pfluc.npy'
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
		if os.path.isfile(pfluc_path):
			data = np.load(pfluc_path)
			print(pfluc_path,'already exists')
			break

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

np.save(pfluc_path, data)

# ------------------------------
# Boundary Layer thickness calculation
# ------------------------------
#Read the fluctuation of tangential velocity
if if_read_boundary_velocity == True:
	bl_var = 'U_t'
	ut_fluc_path = bl_read_path + 'ut_fluc.npy'
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
			if os.path.isfile(ut_fluc_path):
				ut_data = np.load(ut_fluc_path)
				print(ut_fluc_path,'already exists')
				break

		for n,i in enumerate(BL_line_prof[0].keys()[1:]):
			for m in range(len(xcoor)):  # read all spatial locations in the current timestep
				profile_append = np.array(BL_line_prof[0][i][bl_var][m])
				if (m==0):
					profile = profile_append #Create the data to be appended by concatenating the wall normal profiles for each x location
				elif (m==1):
					profile = np.concatenate((profile[:,np.newaxis], profile_append[:,np.newaxis]), axis=1)
				else :
					profile = np.concatenate((profile, profile_append[:,np.newaxis]), axis=1)

			ut_data_append = profile
			if (n == 0) and (j == 0):
				ut_data = ut_data_append
			elif (n == 1) and (j == 0):
				ut_data = np.concatenate((ut_data[np.newaxis,:,:], ut_data_append[np.newaxis,:,:]), axis=0)
			else:
				ut_data = np.concatenate((ut_data, ut_data_append[np.newaxis,:,:]), axis=0)

		print('data shape is {}'.format(np.shape(ut_data)))
		print('chunk {} read'.format(j))

	np.save(ut_fluc_path, ut_data)

	#Obtain the mean tangential velocity
	ut_mean = ut_data.mean(axis=0,dtype=np.float64)

# ------------------------------
# Cross-correlation  calculation
# ------------------------------

nbi = data.shape[0]
meandata = data.mean(axis=0,dtype=np.float64)

#Set the masks
h_start = 0.00*delta_95 #start location of the fixed point
h_end = 1.0*delta_95 #end location of the fixed point
h_mask_plot_range = (hcoor > h_start) & (hcoor < h_end) #Create mask for wall normal extent of the plot
h_masked = hcoor[h_mask_plot_range] # Obtain wall normal coordinate masked to plot range of L22

#Setting the fixed point
l0 = analysis.find_nearest(hcoor,temporal.h_0_bar*delta_95) #wall normal coordinate of fixed point
ki0 = analysis.find_nearest(xcoor,xcoor0) #streamwise coordinate of fixed point
h_mask_delta_95 = (hcoor < delta_95) #Mask to scope out the boundary layer

#Display boundary layer thickness
if if_read_boundary_velocity:
	delta_95_calc = extract_BL_params.get_delta95(hcoor,ut_mean[:,ki0])

#Compute the arithmetic mean along the specified axis.
pfluc = data - np.tile(meandata,(nbi,1,1))
Rxt_spectrum = np.zeros((hcoor.shape[0],scoor.shape[0]))

#Smooth interpolated error
if if_interpolate == True:
	for ki in range(0,np.shape(pfluc)[2]-1):
		for t in range(0,np.shape(pfluc)[0]-1):
			i_zero = np.where(abs(pfluc[t,h_mask_delta_95,ki]) == 0)[0]
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
		p1 = butter_bandpass_filter(p1, 2500, 8000, fs, order=5)
		p0 = butter_bandpass_filter(p0, 2500, 8000, fs, order=5)
		c = analysis.get_velocity_corr(p0,p1)
		Rxt_spectrum[l,ki] = c

print('shape of the matrix is',pfluc.shape)
print('format is (time,wall normal,streamwise)')

#Define the limits of the plot
S,H =np.meshgrid((scoor-scoor[ki0]),hcoor)
fig,ax = plt.subplots(figsize=(5,8))

# Contour plot with black lines
levels = np.linspace(0.1, 1.0, 9)
CS = ax.contour(S/delta_95, H/delta_95, Rxt_spectrum, levels=levels, colors='black')
plt.clabel(CS, fmt='%1.1f', inline=True, fontsize=10)
ax.set_xlim([-0.35, 0.35])
ax.set_ylim([0, 2.0]) 
ax.set_xlabel(r'$X/delta^{95}$', fontsize=22)
ax.set_ylabel(r'$H/delta^{95}$', fontsize=22)
interval = 20
levels = np.linspace(-25, 25, 51)
plt.tight_layout()
plt.savefig(probe_save_path + 'velocity_corr_contour_0p{}'.format(int(temporal.h_0_bar*100)))
Rxt_spectrum = []

#Calculate integral length scale 22+
if if_integrate_axis == True:
	integration_axis_list = ['column']
	direction_list = ['plus']
	for i,integration_axis in enumerate(integration_axis_list):
		L_22, scale = analysis.exp_fit_length_scale(pfluc,scoor,hcoor,scoor[ki0],h_start,scoor[ki0],h_end,fs,delta_95,axis=integration_axis,direction = direction_list[i])
		L_22_df = pd.DataFrame({'wall distance': scale, 'L22+':L_22})
		L_22_df.to_csv(probe_save_path + 'L22_{}_{}'.format(direction_list[i],integration_axis),index=False)

#Calculate integral length scale 22+ field

if if_integrate_field == True:
	bl_param = pd.read_csv(bl_save_path + '{}_surface_parameter.csv'.case_name)
	L_22_dict = {} # Initialize dict to store L22 bl_param field
	# Compute the integral length scale along the wall normal direction for each streamwise point
	for i,x0_aux in enumerate(xcoor):
		L_22_aux, _ = analysis.exp_fit_length_scale(pfluc,scoor,hcoor,scoor[i],h_start,scoor[i],h_end,axis='column',direction = 'plus')
		delta_95 = bl_param['delta_95'][i]
		L_22_dict['hcoor/delta_95'] = hcoor/delta_95
		L_22_dict['L22/delta_95'] = L_22_aux/delta_95
		L_22_df.to_csv(bl_save_path + 'L22_profile_{}.csv'.format(str(i).zfill(3)))

#Plot the contour
if troubleshoot == True:
	levels = np.linspace(-3.5, 3.5, 21)  # 20 levels from 0 to 10
	cmap = 'rainbow'
	print('total number of time steps is {}'.format(np.shape(pfluc)[0]))
	for t in range(0,int(np.shape(pfluc)[0]),timestep_interval):
		print('creating contour of timestep {}'.format(t))
		fig,ax = plt.subplots(figsize=(8,8))
		
		CS = ax.contourf(S/0.1356, H/0.1356, pfluc[t,:,:], levels=levels, cmap=cmap)
		ax.set_xlim([-1.0, 0.0])
		ax.set_ylim([0, 0.1]) 
		ax.set_xlabel(r'$X/delta^{95}$', fontsize=22)
		ax.set_ylabel(r'$H/delta^{95}$', fontsize=22)
		# Add a colorbar
		cbar = plt.colorbar(CS, ax=ax)
		cbar.set_label('Velocity', fontsize=18)
		plt.savefig(probe_save_path + 'velocity_contour_timestep_{}'.format(t))
		plt.close()

		plt.plot(hcoor,pfluc[t,:,ki0])
		plt.ylim([-3.0,3.0])
		plt.close()
		plt.savefig(probe_save_path + 'vertical_velocity_profile_timestep_{}'.format(t))
else:
	None