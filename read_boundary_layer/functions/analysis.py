import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
from scipy.integrate import quad
from scipy import interpolate, integrate
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

def get_pearson_corr(signal_1,signal_2,dt):
	#takes in two signals (input as fluctuations about the mean) and outputs the cross-correlation with time lag
	#signal_1 = signal_1 - np.average(signal_1)
	#signal_2 = signal_2 - np.average(signal_1)
	signal_1 = signal_1/(np.std(signal_1)*np.sqrt(len(signal_1)))
	signal_2 = signal_2/(np.std(signal_2)*np.sqrt(len(signal_2)))
	cross_norm = sig.correlate(signal_1, signal_2, mode='full', method='direct')
	time_cross_corr = np.arange(-len(signal_1), len(signal_1)-1)*dt
	return time_cross_corr,cross_norm

def get_velocity_corr(signal_1,signal_2,height_1,height_2):
        #Ensure that signal_1 belongs to the fixed point
        #takes in two signals (input as fluctuations about the mean) and outputs the cross-correlation with time lag
        signal_1 = signal_1 - np.average(signal_1)
        signal_2 = signal_2 - np.average(signal_2)

        if height_2 <= height_1:
            cross_corr_matrix= np.cov(signal_1, signal_2, ddof=0)/(np.std(signal_1)*np.std(signal_1)) #normalize velocity signal
            cross_corr = cross_corr_matrix[0,1]
        else:
            cross_corr_matrix = np.cov(signal_1, signal_2, ddof=0)/(np.std(signal_1)*np.std(signal_2))
            cross_corr = cross_corr_matrix[0,1]
        return cross_corr

###############################################PROCESSING CONTOUR DATA#################################################################
#Obtain the length scale of vertical velocity fluc L22 given the 2D array representing the cross correlation contour
def get_length_scale(pfluc,x,y,x0,y0,x1,y1,threshold = 0.05,axis = 'column'):
	'''
    Computes the integral length scale along a line
	Input
	pfluc: array - 3D array of the time history of data on a plane. 
	x: array - 1D array of horizontal axis
	y: array - 1D array of vertical axis
	x0,y0,x1,y1 : float - x0,y0 is the location of the start of plot range and x1,y1 is the location of the end of plot range
	threshold : float - limit of cross correlation to consider for integration (Default is 0.05)
	axis: str - axis along which to calculate the integral length scale. 'column' or 'row'
	
	Return
	L_scale: array - Computed length scale for each spatial location along the chosen axis
	scale: array - spatial axis along which length scale is computed
	'''

	if axis == 'column':
		ki0 = find_nearest(x,x0) #Find the x coordinate index of the origin
		mask_plot_range = (y > y0) & (y < y1)
		L_scale = np.zeros(len(y[mask_plot_range]))
		for i, y0_i in enumerate(y[mask_plot_range]): #Loop through the fixed point
			mask_integrate_range = (y > y0_i) #Define the integration range for each point to be plotted
			Rxt_spectrum_aux = [] #Declare an array for storing cross corr. on integration axis
			loc_array = []
			#Recompute the cross correlation array
			for j,y_i in enumerate(y[mask_integrate_range][i:]): #moving point
				p1 = pfluc[:,mask_integrate_range,:][:,j,ki0]
				p0 = pfluc[:,mask_plot_range,:][:,i,ki0]
				c = get_velocity_corr(p0,p1,y0_i,y_i)
				if (c > threshold) & (j != len(y[mask_integrate_range])-1):
					Rxt_spectrum_aux.append(c)
					loc_array.append(y_i)
				else:
					break
			L_scale[i] = np.trapz(np.array(Rxt_spectrum_aux),np.array(loc_array-loc_array[0]))
			scale = y[mask_plot_range]
		return L_scale, scale
	
	elif axis == 'row':
		ki0 = find_nearest(x,x0) #Find the x coordinate index of the origin
		mask_plot_range = (y > y0) & (y < y1)
		L_scale = np.zeros(len(y[mask_plot_range]))
		for i, y0_i in enumerate(y[mask_plot_range]): #Loop through the fixed point
			mask_integrate_range = (x > x0) #Define the integration range for each point to be plotted
			Rxt_spectrum_aux = [] #Declare an array for storing cross corr. on integration axis
			loc_array = []
			#Recompute the cross correlation array
			for j,x_i in enumerate(x[mask_integrate_range]): #moving point
				p1 = pfluc[:,mask_plot_range,:][:,i,ki0+j]
				p0 = pfluc[:,mask_plot_range,:][:,i,ki0]
				c = get_velocity_corr(p0,p1,y0_i,y0_i)
				if c > threshold and (j != len(x[mask_integrate_range])-1):
					Rxt_spectrum_aux.append(c)
					loc_array.append(x_i)
				else:
					break
			L_scale[i] = np.trapz(np.array(Rxt_spectrum_aux),np.array(loc_array-loc_array[0]))
			scale = y[mask_plot_range]
		return L_scale, scale
	else :
		print('Invalid choice of axis for length scale calculation')

def interpolate_zeros(array):
    non_zero_indices = np.where(array != 0.0)[0]
    zero_indices = np.where(array == 0.0)[0]

    # Create interpolation function using cubic spline
    interp_func = interpolate.interp1d(non_zero_indices, array[non_zero_indices], kind='cubic', fill_value="extrapolate")

    # Interpolate zero values
    interpolated_array = array.copy()
    interpolated_array[zero_indices] = interp_func(zero_indices)

    return interpolated_array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mov_avg(X,k):
    X_new = X
    for i in range(k//2,X_new.size-k//2):
        X_new[i] = sum(X[i-k//2:i+k//2])/k
    return X_new
