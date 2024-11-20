#This script takes boundary layer time histories of u and v velocity components as input and outputs u_rms and v_rms

from antares import *
from functions import analysis, extract_BL_params
from scipy.signal import savgol_filter
import vtk
import matplotlib.pyplot as plt
import numpy as np
import temporal
import pickle
import os
import math
import h5py
import pandas as pd
import pdb

# ---------------------
# Defined functions
# ---------------------

def fit_and_derivative(x, y, degree=3):
    """
    Fits the given data (x, y) to a polynomial of the specified degree,
    and computes its first derivative.
    
    Parameters:
    - x: Independent variable (input data).
    - y: Dependent variable (data to fit).
    - degree: The degree of the polynomial to fit (default is 1 for linear).
    
    Returns:
    - p: Coefficients of the fitted polynomial.
    - first_derivative: The first derivative of the polynomial at each x.
    """
    # Fit the data to a polynomial of the specified degree
    p = np.polyfit(x, y, degree)  # p contains the polynomial coefficients
    # Create the polynomial from the coefficients
    poly = np.poly1d(p)
    # Compute the first derivative of the polynomial
    poly_derivative = poly.deriv(1)  # First derivative of the polynomial
    # Calculate the values of the first derivative at each x
    first_derivative = poly_derivative(x)
    return p, first_derivative

# ------------------
# Reading the files
# ------------------

settings = pd.read_csv("../setting.csv", index_col= 0)
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness
density = 1.251
Uinf = 16.6
case_name = 'L08'

mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path
bl_save_path = temporal.project_path + 'boundary_layer_profile/'

nb_points = temporal.nb_points #number of points across the boundary layer
var_list = ['U_n','U_t','static_pressure','mag_velocity_rel'] #variable used for the cross correlation contour
timestep_size = temporal.timestep_size
if_interpolate = True
kinematic_viscosity = eval(temporal.kinematic_viscosity)

# Set the total number of timesteps and the number of chunks
step_per_chunk = temporal.step_per_chunk
total_timesteps = temporal.total_timesteps
starting_timestep = temporal.starting_timestep
num_chunks = (total_timesteps - starting_timestep) // step_per_chunk

os.makedirs(bl_save_path, exist_ok=True)

#Read the mesh
r=Reader('hdf_antares')
r['filename'] = mesh_read_path + 'interpolation_3d_grid.h5'
BL_line_geom=r.read()
print('shape of BL line geom',BL_line_geom[0][0]['x'].shape)

print('Loading geometry...')
coordinates_df = {}
coordinates_df['xBL'] = BL_line_geom[0][0]['x'][:,0,0]/0.1356 + 1.0
coordinates_df['yBL'] = BL_line_geom[0][0]['y'][:,0,0]/0.1356
coordinates_df = pd.DataFrame(coordinates_df)
coordinates_df.index.name = 'idxBL'
coordinates_df.to_csv(bl_save_path + '{}_coordinates.csv'.format(case_name))

print('Loading data...')
data_dict = {}

#creation of streamwise distance array
xcoor = BL_line_geom[0][0]['x'][:,0,0]
ycoor = BL_line_geom[0][0]['y'][:,0,0] 
ds = np.sqrt((xcoor[1:]-xcoor[:-1])**2 + (ycoor[1:]-ycoor[:-1])**2)
sprof = np.zeros(ds.size+1,)
sprof[1:] = np.cumsum(ds)
scoor = sprof
hcoor = np.linspace(0,temporal.length_extraction,nb_points)

if os.path.isfile(bl_save_path + 'data_dict.pkl'):
	with open(bl_save_path + 'data_dict.pkl', 'rb') as f:
		data_dict = pickle.load(f)
else:
	#Load the boundary layer data
	#For every timestep form a 2D matrix of velocities (wall normal , streamwise)
	#Array dim : 1 is time, 2 is wall normal and 3 is chordwise
	for j in range(num_chunks):
		#Read the boundary layer history at x_loc
		r = Reader('hdf_antares')
		r['filename'] = bl_read_path + 'BL_line_prof_{}_{}.h5'.format(starting_timestep+j*step_per_chunk,starting_timestep+(j+1)*(step_per_chunk))
		BL_line_prof = r.read()
		for var in var_list:
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
					data_dict[var] = data_append
				elif (n == 1) and (j == 0):
					data_dict[var] = np.concatenate((data_dict[var][np.newaxis,:,:], data_append[np.newaxis,:,:]), axis=0)
				else:
					data_dict[var] = np.concatenate((data_dict[var], data_append[np.newaxis,:,:]), axis=0)
				print('data shape is {}'.format(np.shape(data_dict[var])))
				print('chunk {} read'.format(j))

	for var in var_list:
		if if_interpolate == True:
			for ki in range(0,np.shape(data_dict[var])[2]-1):
				for t in range(0,np.shape(data_dict[var])[0]-1):
					i_zero = np.where(abs(data_dict[var][t,:,ki]) == 0)[0]
					for i in i_zero:
						if i+1 not in i_zero:
							data_dict[var][t,i,ki] = (data_dict[var][t,i-1,ki] + data_dict[var][t,i+1,ki])/2
						else:
							#Obtain the next non-zero value for interpolation
							n=0
							while (data_dict[var][t,i+n,ki]==0):
								n+=1
							data_dict[var][t,i,ki] = (data_dict[var][t,i-1,ki] + data_dict[var][t,i+n,ki])/2
	# Save the boundary layer data
	with open(bl_save_path + 'data_dict.pkl', 'wb') as f:
		pickle.dump(data_dict, f)
# ------------------------------
# Parameter calculation
# ------------------------------

# Compute reynolds stress
data_dict['Ut_mean'] = data_dict['U_t'].mean(axis=0,dtype=np.float64)
data_dict['Un_mean'] = data_dict['U_n'].mean(axis=0,dtype=np.float64)

nbi = data_dict['static_pressure'].shape[0] # number of instants
#Compute the arithmetic mean along the specified axis.
data_dict['Ut_fluc'] = data_dict['U_t'] - np.tile(data_dict['Ut_mean'],(nbi,1,1))
data_dict['Un_fluc'] = data_dict['U_n'] - np.tile(data_dict['Un_mean'],(nbi,1,1))

data_dict['uu_mean'],data_dict['vv_mean'],data_dict['uv_mean'] = np.zeros((hcoor.shape[0],scoor.shape[0])),np.zeros((hcoor.shape[0],scoor.shape[0])),np.zeros((hcoor.shape[0],scoor.shape[0]))
#Define the x and y coord array
S,H =np.meshgrid(scoor,hcoor)

print('Computing reynolds stress in the boundary layer...')

#Calculate the rms for each point in space
for istreamwise in range(0,np.shape(data_dict['uv_mean'])[1]):                                          #streamwise index_point
	for iwallnormal in range(0,np.shape(data_dict['uv_mean'])[0]):                                       #wall normal index_point
		U_n = data_dict['Un_fluc'][:,iwallnormal,istreamwise]
		U_t = data_dict['Ut_fluc'][:,iwallnormal,istreamwise]
		data_dict['uu_mean'][iwallnormal,istreamwise],data_dict['vv_mean'][iwallnormal,istreamwise],data_dict['uv_mean'][iwallnormal,istreamwise] = analysis.get_velocity_cov(U_t,U_n)

# Compute delta_95, momentum thickness and displacement thickness
delta_95, delta_theta, delta_star, beta_c, RT, cf, uv_max, Ue, tau_wall, edge_pressure, y_w = tuple(np.zeros(len(scoor)) for _ in range(11))
data_dict['static_pressure_mean'] = data_dict['static_pressure'].mean(axis=0,dtype=np.float64)
data_dict['mag_velocity_rel_mean'] = data_dict['mag_velocity_rel'].mean(axis=0,dtype=np.float64)

print('Computing parameter of the boundary layer...')

for istreamwise,streamwise_coor in enumerate(scoor):
	data_dict['Ut_mean'][:,istreamwise][0] = 0.0 # Enforce 1st element velocity to be zero
	data_dict['mag_velocity_rel_mean'][:,istreamwise][0] = 0.0 # Enforce 1st element velocity to be zero
	U_t = data_dict['Ut_mean'][:,istreamwise]
	mag_velocity_rel = data_dict['mag_velocity_rel_mean'][:,istreamwise]
	total_pressure = data_dict['static_pressure_mean'][:,istreamwise] + 0.5*density*(data_dict['mag_velocity_rel_mean'][:,istreamwise]**2)
	total_pressure = total_pressure - total_pressure[0]
	dudy = np.zeros(np.size(mag_velocity_rel)-1)
	dudy = np.diff(U_t)/np.diff(hcoor)
	dudy_interp = np.interp(hcoor,hcoor[:-1]+np.diff(hcoor)/2,dudy)

	idx_delta_95,delta_95[istreamwise] = extract_BL_params.get_delta95(hcoor,total_pressure)
	uv_max[istreamwise] = np.max(abs(density*data_dict['uv_mean'][:,istreamwise]))
	Ue[istreamwise] = U_t[idx_delta_95]
	q = 0.5*density*mag_velocity_rel[idx_delta_95]**2
	delta_star[istreamwise],delta_theta[istreamwise] = extract_BL_params.get_boundary_layer_thicknesses_from_line(hcoor,U_t,density,idx_delta_95)
	tau_wall[istreamwise] = abs((U_t[1] - U_t[0])/(hcoor[1]-hcoor[0])*kinematic_viscosity)
	edge_pressure[istreamwise] = data_dict['static_pressure_mean'][idx_delta_95,istreamwise]

	u_tau = np.sqrt(tau_wall[istreamwise]/density)
	cf[istreamwise] = tau_wall[istreamwise]/q
	RT[istreamwise] = u_tau*delta_95[istreamwise]/kinematic_viscosity*np.sqrt(cf[istreamwise]/2)

	#Obtain the parameters for Pargal model
	y_plus = hcoor*u_tau/kinematic_viscosity
	u_plus = U_t/u_tau
	if istreamwise > len(scoor)//2: #APG
		kappa = 0.3
		B = -1.38
	else: #ZPG
		kappa = 0.41
		B = 4.5
	D = u_plus - (1/kappa*np.log(y_plus)+B) # Compute the diagnostic function
	y_idx = np.where(abs(D) < 1.0)[0][-1]
	y_w[istreamwise] = y_plus[y_idx]

	bl_data = pd.DataFrame({
		'h' : hcoor,
		'U_t' : U_t,
		'dudy' : dudy_interp,
		'uu_mean' : data_dict['uu_mean'][:,istreamwise],
		'vv_mean' : data_dict['vv_mean'][:,istreamwise],
		'uv_mean' : data_dict['uv_mean'][:,istreamwise]
	})
	bl_data.to_csv(bl_save_path + '{}_BL_{}.csv'.format(case_name,str(istreamwise).zfill(3)))

print('Computing pressure gradient...')
smoothed_static_pressure = savgol_filter(edge_pressure[:-1], window_length=11, polyorder=2)
dpds = np.zeros(np.size(smoothed_static_pressure)-1)
dpds = np.diff(smoothed_static_pressure)/np.diff(scoor[:-1])
dpds_interp = np.interp(scoor,scoor[:-2],dpds)
data_dict['dpds'] = dpds_interp
beta_c = delta_theta/tau_wall*data_dict['dpds']

# Save boundary layer info
surface_data = pd.DataFrame({
	'streamwise location' : scoor,
    'delta_95': delta_95,
    'delta_theta': delta_theta,
    'delta_star': delta_star,
    'beta_c': beta_c,
    'RT': RT,
	'dpds' : data_dict['dpds'],
	'cf' : cf,
	'tau_wall' : tau_wall,
	'Ue' : Ue,
	'uv_max' : uv_max,
	'y_w' : y_w
})

# Smooth profile in streamwise direction
smooth_var_list = ['beta_c','dpds','cf','tau_wall','Ue','uv_max']
for smooth_var in smooth_var_list:
	surface_data[smooth_var][:-1] = savgol_filter(surface_data[smooth_var][:-1], 11, 2)

surface_data.to_csv(bl_save_path + '{}_surface_parameter.csv'.format(case_name))
