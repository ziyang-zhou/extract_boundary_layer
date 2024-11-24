#This script takes boundary layer time histories of u and v velocity components as input and outputs u_rms and v_rms

from antares import *
from functions import analysis, extract_BL_params
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit
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

def Ut_function(hcoor,tau_wall,offset,density=1.25,kinematic_viscosity=1.44e-5):
	Ut = hcoor*tau_wall/kinematic_viscosity/density+offset
	return Ut

# ------------------
# Reading the files
# ------------------

settings = pd.read_csv("../setting.csv", index_col= 0)
delta_95 = eval(settings.at["delta_95", settings.columns[0]]) #Read the boundary layer thickness
density = 1.251
Uinf = 16.6
case_name = 'L08'
chord = 0.1356

mesh_read_path = temporal.mesh_path
bl_read_path = temporal.bl_path
bl_save_path = temporal.project_path + 'boundary_layer_profile/'

wall_shear_method = 'shear_fit' #smoothed_derivative or spline or shear_fit

nb_points = temporal.nb_points #number of points across the boundary layer
var_list = ['U_n','U_t','static_pressure','mag_velocity_rel'] #variable used for the cross correlation contour
timestep_size = temporal.timestep_size
if_interpolate = temporal.if_interpolate
kinematic_viscosity = eval(temporal.kinematic_viscosity)

# Set the total number of timesteps and the number of chunks
step_per_chunk = temporal.step_per_chunk
total_timesteps = temporal.total_timesteps
starting_timestep = temporal.starting_timestep
num_chunks = (total_timesteps - starting_timestep) // step_per_chunk

os.makedirs(bl_save_path, exist_ok=True)
os.makedirs(bl_save_path + 'FIG/', exist_ok=True)

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
			idx_start = 1 if step_per_chunk > 1 else 0
			for n,i in enumerate(BL_line_prof[0].keys()[idx_start:]):
				for m in range(len(xcoor)):  # read all spatial locations in the current timestep
					profile_append = np.array(BL_line_prof[0][i][var][m])
					if (m==0):
						profile = profile_append #Create the data to be appended by concatenating the wall normal profiles for each x location
					elif (m==1):
						profile = np.concatenate((profile[:,np.newaxis], profile_append[:,np.newaxis]), axis=1)
					else :
						profile = np.concatenate((profile, profile_append[:,np.newaxis]), axis=1)
				data_append = profile
				print('n {} j {}'.format(n,j))
				if (n + j == 0):
					data_dict[var] = data_append
				elif (n + j == 1):
					data_dict[var] = np.concatenate((data_dict[var][np.newaxis,:,:], data_append[np.newaxis,:,:]), axis=0)
				else:
					data_dict[var] = np.concatenate((data_dict[var], data_append[np.newaxis,:,:]), axis=0)
				print('data shape is {}'.format(np.shape(data_dict[var])))
				print('chunk {} read'.format(j))
	# Save the boundary layer data
	with open(bl_save_path + 'data_dict.pkl', 'wb') as f:
		pickle.dump(data_dict, f)

for var in var_list:
	if if_interpolate == True:
		if data_dict[var].ndim == 3:
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
		else: # variable is mean
			for ki in range(0,np.shape(data_dict[var])[1]-1):
					i_zero = np.where(abs(data_dict[var][:,ki]) == 0)[0]
					for i in i_zero:
						if i+1 not in i_zero:
							data_dict[var][i,ki] = (data_dict[var][i-1,ki] + data_dict[var][i+1,ki])/2
						else:
							#Obtain the next non-zero value for interpolation
							n=0
							while (data_dict[var][i+n,ki]==0):
								n+=1
							data_dict[var][i,ki] = (data_dict[var][i-1,ki] + data_dict[var][i+n,ki])/2			


# ------------------------------
# Parameter calculation
# ------------------------------
if data_dict['static_pressure'].ndim == 2:
	nbi = 0 # this means only the mean frame is available
else:
	nbi = data_dict['static_pressure'].shape[0] # number of instants

if nbi is not 0:
	# Compute reynolds stress
	data_dict['Ut_mean'] = data_dict['U_t'].mean(axis=0,dtype=np.float64)
	data_dict['Un_mean'] = data_dict['U_n'].mean(axis=0,dtype=np.float64)
	#Compute the arithmetic mean along the specified axis.
	data_dict['Ut_fluc'] = data_dict['U_t'] - np.tile(data_dict['Ut_mean'],(nbi,1,1))
	data_dict['Un_fluc'] = data_dict['U_n'] - np.tile(data_dict['Un_mean'],(nbi,1,1))
	data_dict['uu_mean'],data_dict['vv_mean'],data_dict['uv_mean'] = np.zeros((hcoor.shape[0],scoor.shape[0])),np.zeros((hcoor.shape[0],scoor.shape[0])),np.zeros((hcoor.shape[0],scoor.shape[0]))
	data_dict['static_pressure_mean'] = data_dict['static_pressure'].mean(axis=0,dtype=np.float64)
	data_dict['mag_velocity_rel_mean'] = data_dict['mag_velocity_rel'].mean(axis=0,dtype=np.float64)
else:
	data_dict['Ut_mean'] = data_dict['U_t']
	data_dict['Un_mean'] = data_dict['U_n']
	data_dict['Ut_fluc'] = np.ones((5,hcoor.shape[0],scoor.shape[0]))
	data_dict['Un_fluc'] = np.ones((5,hcoor.shape[0],scoor.shape[0]))
	data_dict['uu_mean'],data_dict['vv_mean'],data_dict['uv_mean'] = np.ones((hcoor.shape[0],scoor.shape[0])),np.ones((hcoor.shape[0],scoor.shape[0])),np.ones((hcoor.shape[0],scoor.shape[0]))
	data_dict['static_pressure_mean'] = data_dict['static_pressure']
	data_dict['mag_velocity_rel_mean'] = data_dict['mag_velocity_rel']

#Define the x and y coord array
S,H =np.meshgrid(scoor,hcoor)
print('Computing reynolds stress in the boundary layer...')
#Calculate the rms for each point in space
for istreamwise in range(0,np.shape(data_dict['uv_mean'])[1]):                                          #streamwise index_point
	for iwallnormal in range(0,np.shape(data_dict['uv_mean'])[0]):                                       #wall normal index_point
		U_n = data_dict['Un_fluc'][:,iwallnormal,istreamwise]
		U_t = data_dict['Ut_fluc'][:,iwallnormal,istreamwise]
		data_dict['uu_mean'][iwallnormal,istreamwise],data_dict['vv_mean'][iwallnormal,istreamwise],data_dict['uv_mean'][iwallnormal,istreamwise], = analysis.get_velocity_cov(U_t,U_n)

# Compute delta_95, momentum thickness and displacement thickness
delta_95, delta_theta, delta_star, beta_c, RT, cf, uv_max, Ue, tau_wall, edge_pressure, y_w, p_rms = tuple(np.zeros(len(scoor)) for _ in range(12))


print('Computing parameter of the boundary layer...')

for istreamwise,streamwise_coor in enumerate(scoor):
	data_dict['Ut_mean'][:,istreamwise][0] = 0.0 # Enforce 1st element velocity to be zero
	data_dict['mag_velocity_rel_mean'][:,istreamwise][0] = 0.0 # Enforce 1st element velocity to be zero
	hcoor[0] = 0.0
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

	if wall_shear_method == 'spline': 	
		tau_spl = CubicSpline(hcoor[1:], U_t[1:], bc_type = 'natural')
		dudy_wall = tau_spl.c[-2,0]
		tau_wall[istreamwise] = dudy_wall*kinematic_viscosity*density
	elif wall_shear_method == 'legacy_spline':
		cs = CubicSpline(hcoor,U_t)
		x_0 = 0
		dudy_wall = cs(x_0, 1)
		tau_wall[istreamwise] = dudy_wall*kinematic_viscosity*density
	elif wall_shear_method == 'smoothed_derivative':
		tau_wall[istreamwise] = extract_BL_params.get_wall_shear_stress_from_line(hcoor,U_t,density,kinematic_viscosity,filter_size_var=3,filter_size_der=3,npts_interp=100,maximum_stress=False)
	elif wall_shear_method == 'shear_fit':
		params, _ = curve_fit(Ut_function, hcoor[1:3], U_t[1:3], p0=[1.0,0.3],kinematic_viscosity=kinematic_viscosity,density=density)
		tau_wall[istreamwise] = params[0]

	u_tau_aux = np.sqrt(tau_wall[istreamwise]/density)

	if istreamwise%10 == 0:
		plt.scatter(hcoor[:]*u_tau_aux/kinematic_viscosity,U_t[:]/u_tau_aux,label='data')
		plt.plot(np.linspace(0,5,1000),np.linspace(0,5,1000),label='y+ = u+')
		plt.xlabel('y+')
		plt.ylabel('U+')
		plt.xlim([0.1,100])
		plt.xscale('log')
		plt.legend()
		plt.savefig(bl_save_path + 'FIG/log_law_check_{}.jpg'.format(istreamwise))
	
	#tau_wall[istreamwise] = extract_BL_params.get_wall_shear_stress_from_line(hcoor,U_t,density,kinematic_viscosity,npts_interp=50000,maximum_stress=False)
	#tau_wall[istreamwise] = U_t[1]/hcoor[1]*kinematic_viscosity*density
	edge_pressure[istreamwise] = data_dict['static_pressure_mean'][idx_delta_95,istreamwise]

	u_tau = np.sqrt(tau_wall[istreamwise]/density)
	cf[istreamwise] = tau_wall[istreamwise]/q
	#RT[istreamwise] = u_tau*delta_95[istreamwise]/kinematic_viscosity*np.sqrt(cf[istreamwise]/2)
	RT[istreamwise] = (delta_95[istreamwise]/Ue[istreamwise])/(kinematic_viscosity/u_tau**2)

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

	if not np.any(np.isnan(D)) and not np.any(np.isinf(D)):
		y_idx = np.where(abs(D) < 10.0)[0][-1]
		y_w[istreamwise] = y_plus[y_idx]

	bl_data = pd.DataFrame({
		'h' : hcoor,
		'U_t' : U_t,
		'mag_velocity_rel_mean' : mag_velocity_rel,
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

## Smooth derivative
#from scipy import ndimage
#dP=ndimage.gaussian_filter1d(edge_pressure[:-1],sigma=10, order=1, mode='nearest')
#ds=np.diff(scoor)
#dpds = dP/ds
#dpds_interp = np.interp(scoor,scoor[:-1],dpds)
#data_dict['dpds'] = dpds_interp

beta_c = delta_theta/tau_wall*data_dict['dpds']

# Save boundary layer info
surface_data = pd.DataFrame({
	'streamwise location' : xcoor + chord,
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
