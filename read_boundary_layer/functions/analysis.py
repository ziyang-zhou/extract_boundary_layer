import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
from scipy.integrate import quad
from functions import extract_BL_params
import math
import os
import glob
import pdb
from antares import *

#find the next greatest power of 2 given x
def next_greater_power_of_2(x):  
    return 2**(x-1).bit_length()

#integrate f(x) with respect to x where f(x) is given by arr and dx is the interval of x
def trapezoidal_integration(arr, dx):
    n = arr.shape[0] - 1  # Number of intervals
    integral = 0.0
    for i in range(n):
        integral += (dx / 2) * (arr[i] + arr[i + 1])
    return integral

#defines a gaussian decay function 'func' given x0 (the centroid), A (the amplitude) and sigma (the spread)
def func(x, A, x0, sigma):
    exponent = -(x - x0) ** 2 / (2 * sigma ** 2)
    func_values = A * np.exp(exponent)
    max_value = np.max(func_values)
    return func_values / max_value

#find the index of the value in an array which is closest to given value
def find_closest_index(arr, target):
    closest_index = 0  # Initialize the index of the closest value
    min_difference = abs(arr[0] - target)  # Initialize the minimum absolute difference

    for i in range(1, len(arr)):
        difference = abs(arr[i] - target)
        if difference < min_difference:
            min_difference = difference
            closest_index = i

    return closest_index

def max_norm(x):
    '''
    Returns the min-max (0 -> 1) scaled version of input series x.

    Params:
        x:      list - the data to be scaled.
    Returns
        y:      list - The normalized data.
    '''

    y = x/np.max(x)
    return y


def mask_variables(variables_list, keep_time):
	# takes a list of variables and filters them based on a boolean mask
    filtered_variables = [var[keep_time_mask] for var in variables_list]
    return filtered_variables

##################################################################################ANTARES (BOUNDARY LAYER READING)################################################################################
# Define a function to process a single chunk of data. The function reads the data from b_vol along a line to give boundary layer profile
def process_chunk(chunk_start, chunk_end, density, read_path, save_path):
    r = Reader('hdf_antares')
    r['filename'] = read_path + 'b_vol_{}_{}.h5'.format(chunk_start, chunk_end)
    b_vol = r.read()
    print('zones in b_vol', b_vol.keys())
    print('instants in b_vol', b_vol[0].keys())
    print('variables in each instant of b_vol', b_vol[0][0].keys())
    BL_line_prof, successful_extraction = functions.extract_BL_params.extract_BL_profiles(
        b_vol, BL_line_geom, length_extraction, var_detection, nb_points,
        axis, axis_direction, relative_velocity_vec, density,
        laminar_dynamic_viscosity='0.000015'
    )
    writer = Writer('hdf_antares')
    writer['filename'] = save_path + 'BL_line_prof_{}_{}'.format(chunk_start, chunk_end)
    writer['base'] = BL_line_prof
    writer.dump()

def cf_extraction(num_chunks,step_per_chunk,starting_timestep,total_timesteps,bl_read_path,mesh_read_path):
    # num_chunks : the number of chunks which the time axis of the data has been split into
    # step_per_chunk : number of timesteps in each chunk
    # read_path : directory to read the boundary layer profile from
    # read boundary layer geometry again to obtain the x coord of each BL profile
    r = Reader('hdf_antares')
    r['filename'] = mesh_read_path + 'interpolation_3d_grid.h5'
    BL_line_geom = r.read()

    for j in range(num_chunks): #read the current chunk (of timesteps)
        # read boundary layer data
        r = Reader('hdf_antares')
        r['filename'] = bl_read_path + 'BL_line_prof_{}_{}.h5'.format(starting_timestep + step_per_chunk * j, starting_timestep + (step_per_chunk * (j + 1)))
        BL_line_prof = r.read()
        nbpoints = len(BL_line_geom[0][0]['x'][:, 0, 0])
        ntimesteps = total_timesteps
        nbread = len(BL_line_geom[0][0]['x'][0:nbpoints, 0, 0])
        if j == 0:  # initialize the wall shear matrix
            cf = np.zeros((ntimesteps-starting_timestep, nbread))
            x_coord = BL_line_geom[0][0]['x'][0:nbpoints, 0, 0]
        for n in range(starting_timestep + step_per_chunk * j, starting_timestep + (step_per_chunk * (j + 1))):  # read all timesteps in the current chunk
            for i in range(nbread):  # read all spatial locations in the current timestep
                wall_distance = BL_line_prof[0]['{:04d}'.format(n)]['h'][i]
                relative_velocity_magnitude = BL_line_prof[0]['{:04d}'.format(n)]['U_t'][i]
                U_x = BL_line_prof[0]['{:04d}'.format(n)]['x_velocity'][i]
                signs = np.sign(BL_line_prof[0]['{:04d}'.format(n)]['x_velocity'][i][1])
                density = BL_line_prof[0]['{:04d}'.format(n)]['density'][i]
                kinematic_viscosity = BL_line_prof[0]['{:04d}'.format(n)]['nu_lam'][i]
                tau_wall = extract_BL_params.get_wall_shear_stress_from_line(wall_distance,relative_velocity_magnitude,density,kinematic_viscosity,filter_size_var=3,filter_size_der=3, npts_interp=100,maximum_stress=False)
               
                tau_wall = tau_wall * signs
                if tau_wall == 0.0 or math.isnan(tau_wall):
                    print('wall shear is', tau_wall, 'at', x_coord[i], 'and index', i)
                    print('velocity profile is', relative_velocity_magnitude)
                cf[n-starting_timestep, i] = tau_wall
        del r
        del BL_line_prof
    # Transpose the cf array to have x_coord as the first row
    cf_with_x_coord = np.vstack((x_coord[0:nbpoints], cf))
    return cf_with_x_coord

def mask_suction_side(cf,cf_data):
    '''
    Mask to order to keep the suction side of the airfoil (assuming 1st column is data at TE and 2nd column is on the pressure side)
    Params:
        cf:      array - matrix of cf values to be masked (cf here refers to wall shear in Pa)
                        first row of the matrix contains the x coordinate
    Returns
        cf:      array - returns the same cf matrix with only suction side values
    '''

    num_rows, num_columns = cf.shape
    for i in range(num_columns - 1):
        dx = cf_data[0, i + 1] - cf_data[0, i]
        if dx > 0:
            index = i + 2
            break
    cf = cf[:, index:]
    x_coord = cf_data[0, index:]
    return cf,x_coord

def reattachment_location(cf,x_coord):
    '''
    Calculate the reattachment location by returning the location of last sign change in wall shear, moving from LE to TE edge
    Params:
        cf:      array - matrix of cf values to calculate the reattachment location from
        x_coord: array - x location 
    Returns
        x_attach:      array - returns the x location of the reattachment
    '''
    # Create an array to store the results
    num_rows,num_columns = cf.shape
    print('x_coord',x_coord)
    x_attach = np.zeros(num_rows)
    last_sign_switch_column = np.zeros(num_rows, dtype=int)
    # Iterate through each row, obtaining the final sign change as the definition of the reattachment point.
    for i in range(num_rows): #looping through timestep
        row = cf[i,:]
        # Initialize variables to keep track of the current sign
        current_sign = np.sign(row[0])
        # Iterate through the columns
        for j in range(1, num_columns):
            new_sign = np.sign(row[j])
            # Check if there is a sign switch
            if new_sign != current_sign and not math.isnan(row[j]):
                last_sign_switch_column[i] = j
                x_attach[i] = x_coord[j]
                current_sign = new_sign
                print('index',i,'attach',x_attach[i])
    print('x_attach',x_attach)           
    return x_attach
