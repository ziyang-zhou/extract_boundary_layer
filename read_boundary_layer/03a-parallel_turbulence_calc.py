#This script takes boundary layer time histories of u and v velocity components as input and outputs u_rms and v_rms

from antares import *
from functions import analysis, extract_BL_params
from scipy.signal import savgol_filter
import vtk
import matplotlib.pyplot as plt
import concurrent.futures
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

# Function to process each chunk for a variable (this will be parallelized)
def process_chunk(j, var, starting_timestep, step_per_chunk, BL_line_geom, bl_read_path, xcoor, ycoor, if_interpolate):
	# Read the boundary layer history at x_loc for chunk j
	r = Reader('hdf_antares')
	r['filename'] = bl_read_path + f'BL_line_prof_{starting_timestep + j*step_per_chunk}_{starting_timestep + (j + 1) * step_per_chunk}.h5'
	BL_line_prof = r.read()

	# Process the streamwise distance array only once (for the first chunk)
	if j == 0:
		ds = np.sqrt((xcoor[1:] - xcoor[:-1])**2 + (ycoor[1:] - ycoor[:-1])**2)
		sprof = np.zeros(ds.size + 1)
		sprof[1:] = np.cumsum(ds)
		scoor = sprof
		hcoor = BL_line_prof[0][0]['h'][0, :]  # Read the wall normal distance

	chunk_data = []

	for n, i in enumerate(BL_line_prof[0].keys()[1:]):
		profile = None
		for m in range(len(xcoor)):  # Read all spatial locations in the current timestep
			profile_append = np.array(BL_line_prof[0][i][var][m])
		if (m==0):
			profile = profile_append #Create the data to be appended by concatenating the wall normal profiles for each x location
		elif (m==1):
			profile = np.concatenate((profile[:,np.newaxis], profile_append[:,np.newaxis]), axis=1)
		else :
			profile = np.concatenate((profile, profile_append[:,np.newaxis]), axis=1)

        # Append the profile data to chunk_data
		chunk_data.append(profile)

	# Stack the profiles for each timestep into a single array
	data_append = np.stack(chunk_data, axis=0)

	# Interpolation if required
	if if_interpolate:
		data_append = interpolate_data(data_append)

	return data_append

# Interpolation function
def interpolate_data(data):
    if data is not None:
        for ki in range(0, np.shape(data)[2] - 1):
            for t in range(0, np.shape(data)[0] - 1):
                i_zero = np.where(abs(data[t, :, ki]) == 0)[0]
                for i in i_zero:
                    if i + 1 not in i_zero:
                        data[t, i, ki] = (data[t, i - 1, ki] + data[t, i + 1, ki]) / 2
                    else:
                        # Obtain the next non-zero value for interpolation
                        n = 0
                        while data[t, i + n, ki] == 0:
                            n += 1
                        data[t, i, ki] = (data[t, i - 1, ki] + data[t, i + n, ki]) / 2
    return data

# Function to process each variable (parallelized)
def process_variable(var, num_chunks, starting_timestep, step_per_chunk, BL_line_geom, bl_read_path, xcoor, ycoor, if_interpolate):
    data = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        # Parallelize the chunk processing for each variable
        for j in range(num_chunks):
            futures.append(executor.submit(process_chunk, j, var, starting_timestep, step_per_chunk, BL_line_geom, bl_read_path, xcoor, ycoor, if_interpolate))

        # Collect results from all chunks
        chunk_results = []
        for future in concurrent.futures.as_completed(futures):
            chunk_results.append(future.result())

    # Combine the results from all chunks
    data = np.concatenate(chunk_results, axis=0)

    return data

# Main loop for parallel processing
def parallel_processing(var_list, num_chunks, starting_timestep, step_per_chunk, BL_line_geom, bl_read_path, bl_save_path, if_interpolate=True):
    data_dict = {}

    # Process the streamwise distance array and coordinates only once
    xcoor = BL_line_geom[0][0]['x'][:, 0, 0]
    ycoor = BL_line_geom[0][0]['y'][:, 0, 0]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        # Parallelize the processing of variables
        for var in var_list:
            futures.append(executor.submit(process_variable, var, num_chunks, starting_timestep, step_per_chunk, BL_line_geom, bl_read_path, xcoor, ycoor, if_interpolate))

        # Collect results from all variables
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            var = var_list[futures.index(future)]  # Get the variable associated with the result
            data_dict[var] = result

    # Save the data to a pickle file
    with open(bl_save_path + 'data_dict.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    return data_dict


# ------------------
# Reading the files
# ------------------

settings = pd.read_csv("../setting.csv", index_col= 0)
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness
density = 1.251
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
data_dict = parallel_processing(var_list, num_chunks, starting_timestep, step_per_chunk, BL_line_geom, bl_read_path, bl_save_path, if_interpolate=True)
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
delta_95, delta_theta, delta_star, beta_c, RT, cf, uv_max, Ue, tau_wall = tuple(np.zeros(len(scoor)) for _ in range(9))
data_dict['static_pressure_mean'] = data_dict['static_pressure'].mean(axis=0,dtype=np.float64)
data_dict['mag_velocity_rel_mean'] = data_dict['mag_velocity_rel'].mean(axis=0,dtype=np.float64)

print('Computing pressure gradient...')
smoothed_static_pressure = savgol_filter(data_dict['static_pressure_mean'][0,:-1], window_length=5, polyorder=2)
dpds = np.zeros(np.size(smoothed_static_pressure)-1)
dpds = np.diff(smoothed_static_pressure)/np.diff(scoor[:-1])
dpds_interp = np.interp(scoor,scoor[:-2],dpds)
data_dict['dpds'] = dpds_interp

print('Computing parameter of the boundary layer...')

for istreamwise,streamwise_coor in enumerate(scoor):
	total_pressure = data_dict['static_pressure_mean'][:,istreamwise] + 0.5*density*(data_dict['mag_velocity_rel_mean'][:,istreamwise]**2)
	total_pressure = total_pressure - total_pressure[0]
	data_dict['Ut_mean'][:,istreamwise][0] = 0.0 # Enforce 1st element velocity to be zero
	data_dict['mag_velocity_rel_mean'][:,istreamwise][0] = 0.0 # Enforce 1st element velocity to be zero
	U_t = data_dict['Ut_mean'][:,istreamwise]
	mag_velocity_rel = data_dict['mag_velocity_rel_mean'][:,istreamwise]
	dudy = np.zeros(np.size(mag_velocity_rel)-1)
	dudy = np.diff(U_t)/np.diff(hcoor)
	dudy_interp = np.interp(hcoor,hcoor[:-1]+np.diff(hcoor)/2,dudy)

	idx_delta_95,delta_95[istreamwise] = extract_BL_params.get_delta95(hcoor,total_pressure)
	uv_max[istreamwise] = np.max(abs(data_dict['uv_mean'][:,istreamwise]))
	Ue[istreamwise] = U_t[idx_delta_95]
	q = 0.5*density*mag_velocity_rel[idx_delta_95]**2
	delta_star[istreamwise],delta_theta[istreamwise] = extract_BL_params.get_boundary_layer_thicknesses_from_line(hcoor,U_t,density,idx_delta_95)
	tau_wall[istreamwise] = abs((U_t[1] - U_t[0])/(hcoor[1]-hcoor[0])*kinematic_viscosity)
	beta_c[istreamwise] = delta_theta[istreamwise]/tau_wall[istreamwise]*data_dict['dpds'][istreamwise]
	u_tau = np.sqrt(tau_wall[istreamwise]/density)
	cf[istreamwise] = tau_wall[istreamwise]/q
	RT[istreamwise] = u_tau*delta_95[istreamwise]/kinematic_viscosity*np.sqrt(cf[istreamwise]/2)
	bl_data = pd.DataFrame({
		'h' : hcoor,
		'U_t' : U_t,
		'dudy' : dudy_interp,
		'uu_mean' : data_dict['uu_mean'][:,istreamwise],
		'vv_mean' : data_dict['vv_mean'][:,istreamwise],
		'uv_mean' : data_dict['uv_mean'][:,istreamwise]
	})
	bl_data.to_csv(bl_save_path + '{}_BL_{}.csv'.format(case_name,str(istreamwise).zfill(3)))

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
	'uv_max' : uv_max
})
surface_data.to_csv(bl_save_path + '{}_surface_parameter.csv'.format(case_name))