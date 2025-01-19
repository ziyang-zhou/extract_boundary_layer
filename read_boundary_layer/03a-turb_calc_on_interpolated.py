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

# ---------------------
# Main
# ---------------------

case_name = 'L08'
project_path = temporal.project_path
bl_save_path = project_path + 'boundary_layer_profile/'
interpolated_path = temporal.interpolated_path
data = h5py.File(interpolated_path+'interp_stats.hdf5', 'r')

for istreamwise,x_coord in enumerate(data['mesh']['x'][:,0]):
    norm_x = data['mesh']['x'][istreamwise,1]-data['mesh']['x'][istreamwise,0]
    norm_y = data['mesh']['y'][istreamwise,1]-data['mesh']['y'][istreamwise,0]
    norm_z = data['mesh']['z'][istreamwise,1]-data['mesh']['z'][istreamwise,0]
    Ux_mean = data['x_velocity'][istreamwise,:]
    Uy_mean = data['y_velocity'][istreamwise,:]
    Uz_mean = data['z_velocity'][istreamwise,:]
    Umag_mean = (Ux_mean**2 + Uy_mean**2)**0.5

    xx_mean = data['x_velocity_2'][istreamwise,:] - data['x_velocity'][istreamwise,:]**2
    yy_mean = data['y_velocity_2'][istreamwise,:] - data['y_velocity'][istreamwise,:]**2
    xy_mean = data['x_velocity_y_velocity'][istreamwise,:] - data['x_velocity'][istreamwise,:]*data['y_velocity'][istreamwise,:]

    n_vec=np.array(norm_x,norm_y,norm_z)
    n_vec=normalize_vector(n_vec)
    theta = np.arctan(norm_x/norm_y)

    h_coord = [(data['mesh']['x'][istreamwise,:]-data['mesh']['x'][istreamwise,0]) + (data['mesh']['y'][istreamwise,:]-data['mesh']['y'][istreamwise,0])]**0.5
    Un_mean = n_vec[0]*Ux_mean + n_vec[1]*Uy_mean
    Ut_mean = ((Ux_mean**2 + Uy_mean**2) - Un_mean**2)**0.5

    uv_mean = (xx_mean - yy_mean)*np.sin(theta)*np.cos(theta) + (np.cos(theta)**2 - np.sin(theta)**2)*xy_mean
    uu_mean = xx_mean*np.cos(theta)**2 - 2*xy_mean*np.sin(theta)*np.cos(theta) + yy_mean*np.sin(theta)**2
    vv_mean = xx_mean*np.sin(theta)**2 + 2*xy_mean*np.sin(theta)*np.cos(theta) + yy_mean*np.cos(theta)**2

	# Smooth derivative
    dU=ndimage.gaussian_filter1d(Ut_mean,sigma=11, order=1, mode='nearest')
    dh=h_coord[1]-h_coord[0]
    dudy_interp = dU/dh

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
