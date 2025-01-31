#This script takes boundary layer time histories of u and v velocity components as input and outputs u_rms and v_rms

from antares import *
from functions import analysis, extract_BL_params
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy import ndimage
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

def normalize_vector(vec):
    magnitude = np.linalg.norm(vec) # Compute the magnitude using NumPy
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vec / magnitude

def Ut_function(hcoor,tau_wall,density=1.25,kinematic_viscosity=1.44e-5):
	Ut = hcoor*tau_wall/kinematic_viscosity/density
	return Ut

def linear_shear_model(hcoor, tau_wall, offset,density=1.25,kinematic_viscosity=1.44e-5):
    m = tau_wall/(density*kinematic_viscosity)
    return m * hcoor + offset

def log_region_finder(y_plus,Ut_plus,nbpts=30000):
	# Find the lower and upper limit y_plus in the log region and mask y+ and U+ accordingly
	y_plus_interp = np.linspace(0,500,nbpts)
	Ut_plus = savgol_filter(Ut_plus, 301, 2)
	Ut_plus_cs = CubicSpline(y_plus,Ut_plus)

	Ut_plus_derivative = Ut_plus_cs(y_plus_interp, 2)
	y_mask = (y_plus_interp < 80) &  (y_plus_interp > 20)
	Ut_plus_derivative_idx = np.where(abs(Ut_plus_derivative[y_mask]) < 3.0)[0][-1]

	mask = (y_plus < y_plus_interp[Ut_plus_derivative_idx]) & (y_plus > 20)
	mask = (y_plus < 1000) & (y_plus > 20)
	print('log region y_plus : {} to {}'.format(y_plus[mask][0],y_plus[mask][-1]))
	return y_plus[mask],Ut_plus[mask]

def log_law_fit(y_plus,kappa,B):
	Ut_plus = 1/kappa*np.log(y_plus)+B
	return Ut_plus

# ---------------------
# Main
# ---------------------

case_name = 'L08'
density = eval(temporal.density)
kinematic_viscosity = eval(temporal.kinematic_viscosity)
project_path = temporal.project_path
bl_save_path = project_path + 'boundary_layer_profile/'
interpolated_path = temporal.interpolated_path
os.makedirs(bl_save_path, exist_ok=True)
data = h5py.File(interpolated_path+'interp_stats.hdf5', 'r')

#creation of streamwise distance array
xcoor = data['mesh']['x'][:,0]
ycoor = data['mesh']['y'][:,0]
ds = np.sqrt((xcoor[1:]-xcoor[:-1])**2 + (ycoor[1:]-ycoor[:-1])**2)
scoor = np.zeros(ds.size+1,)
scoor[1:] = np.cumsum(ds)
chord = 0.1356

# Compute delta_95, momentum thickness and displacement thickness
delta_95, delta_theta, delta_star, beta_c, RT, cf, uv_max, Ue, tau_wall, p_rms, wall_pressure = tuple(np.zeros(len(data['mesh']['x'][:,0])) for _ in range(11))

for istreamwise,x_coord in enumerate(data['mesh']['x'][:,0]):
    norm_x = data['mesh']['x'][istreamwise,1]-data['mesh']['x'][istreamwise,0]
    norm_y = data['mesh']['y'][istreamwise,1]-data['mesh']['y'][istreamwise,0]
    norm_z = data['mesh']['z'][istreamwise,1]-data['mesh']['z'][istreamwise,0]
    Ux_mean = data['x_velocity'][istreamwise,:]
    Uy_mean = data['y_velocity'][istreamwise,:]
    Uz_mean = data['z_velocity'][istreamwise,:]
    Umag_mean = (Ux_mean**2 + Uy_mean**2)**0.5
    static_pressure = data['pressure'][istreamwise,:]

    xx_mean = data['x_velocity_2'][istreamwise,:] - data['x_velocity'][istreamwise,:]**2
    yy_mean = data['y_velocity_2'][istreamwise,:] - data['y_velocity'][istreamwise,:]**2
    xy_mean = data['x_velocity_y_velocity'][istreamwise,:] - data['x_velocity'][istreamwise,:]*data['y_velocity'][istreamwise,:]

    n_vec=np.array([norm_x,norm_y,norm_z])
    n_vec=normalize_vector(n_vec)
    theta = np.arctan(norm_x/norm_y)

    h_coord = ((data['mesh']['x'][istreamwise,:]-data['mesh']['x'][istreamwise,0])**2 + (data['mesh']['y'][istreamwise,:]-data['mesh']['y'][istreamwise,0])**2)**0.5
    Un_mean = n_vec[0]*Ux_mean + n_vec[1]*Uy_mean
    Ut_mean = ((Ux_mean**2 + Uy_mean**2) - Un_mean**2)**0.5

    #impose zero velocity at the wall
    #Ut_mean[0] = 0.0
    #Un_mean[0] = 0.0
    #Umag_mean[0] = 0.0

    uv_mean = (xx_mean - yy_mean)*np.sin(theta)*np.cos(theta) + (np.cos(theta)**2 - np.sin(theta)**2)*xy_mean
    uu_mean = xx_mean*np.cos(theta)**2 - 2*xy_mean*np.sin(theta)*np.cos(theta) + yy_mean*np.sin(theta)**2
    vv_mean = xx_mean*np.sin(theta)**2 + 2*xy_mean*np.sin(theta)*np.cos(theta) + yy_mean*np.cos(theta)**2

    #Boundary layer calc
    total_pressure = static_pressure + 0.5*density*(Umag_mean**2)
    total_pressure = total_pressure - total_pressure[0]
    idx_delta_95,delta_95[istreamwise] = extract_BL_params.get_delta95(h_coord,total_pressure)
    Ue[istreamwise] = Ut_mean[idx_delta_95]
    q = 0.5*density*Umag_mean[idx_delta_95]**2
    delta_star[istreamwise],delta_theta[istreamwise] = extract_BL_params.get_boundary_layer_thicknesses_from_line(h_coord,Ut_mean,density,idx_delta_95)

    #Re stress
    uv_max[istreamwise] = np.max(uv_mean)

    # Wall shear
    #params, _ = curve_fit(linear_shear_model, h_coord[0:6], Ut_mean[0:6], p0=[0.5,0.5])
    params, _ = curve_fit(Ut_function, h_coord[0:5], Ut_mean[0:5], p0=[0.1])
    tau_wall[istreamwise] = params[0]

    #RT
    u_tau = np.sqrt(tau_wall[istreamwise]/density)
    cf[istreamwise] = tau_wall[istreamwise]/q
    RT[istreamwise] = (delta_95[istreamwise]/Ue[istreamwise])/(kinematic_viscosity/u_tau**2)

	# Smooth derivative
    dU=ndimage.gaussian_filter1d(Ut_mean,sigma=11, order=1, mode='nearest')
    dh=h_coord[1]-h_coord[0]
    dudy_interp = dU/dh

    # dpds
    print('Computing pressure gradient...')
    wall_pressure[istreamwise] = static_pressure[0]
    print('Computing pressure gradient...')
    dpds = np.zeros(np.size(wall_pressure[:-1])-1)
    dpds = np.diff(wall_pressure[:-1])/np.diff(scoor[:-1])
    dpds_interp = np.interp(scoor,scoor[:-2],dpds)
    beta_c = delta_theta/tau_wall*dpds_interp

    bl_data = pd.DataFrame({
        'h' : h_coord,
        'U_t' : Ut_mean,
        'mag_velocity_rel_mean' : Umag_mean,
        'dudy' : dudy_interp,
        'uu_mean' : uu_mean,
        'vv_mean' : vv_mean,
        'uv_mean' : uv_mean
    })    
    bl_data.to_csv(bl_save_path + '{}_BL_{}.csv'.format(case_name,str(istreamwise).zfill(3)), index=False)

# Save boundary layer info
surface_data = pd.DataFrame({
	'streamwise_location' : xcoor + chord,
	'delta_95': delta_95,
	'delta_theta': delta_theta,
	'delta_star': delta_star,
	'beta_c': beta_c,
	'RT': RT,
	'wall_pressure': wall_pressure,
	'dpds' : dpds_interp,
	'cf' : cf,
	'tau_wall' : tau_wall,
	'Ue' : Ue,
	'uv_max' : uv_max,
})

surface_data.to_csv(bl_save_path + '{}_surface_parameter.csv'.format(case_name), index=False)

print('Loading geometry...')
coordinates_df = {}
coordinates_df['xBL'] = xcoor/0.1356 + 1.0
coordinates_df['yBL'] = ycoor/0.1356
coordinates_df = pd.DataFrame(coordinates_df)
coordinates_df.index.name = 'idxBL'
coordinates_df.to_csv(bl_save_path + '{}_coordinates.csv'.format(case_name))