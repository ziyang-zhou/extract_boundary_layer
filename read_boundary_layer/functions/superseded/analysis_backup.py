import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.optimize as opt
from scipy.integrate import quad
import os
import glob
from numpy import exp

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
    
#########################################################aeroacoustic function###########################################################################
def rotor_segment_noise(radial_location, angular_speed,number_rotors, airfoil_self_noise, sound_speed,inlet_speed):
	# takes the rotor parameters and calculates the rotor segment broadband noise from the self noise of the airfoil profile
	# airfoil self_noise should contain frequencies in the 1st column and the SPL in the second column
	# observer is assumed to be positioned directly above the hub
    local_speed = angular_speed*radial_location
    inlet_mach = inlet_speed/sound_speed
    def acoustic_integrand(angular_position,airfoil_self_noise,inlet_speed):
        return airfoil_self_noise/((inlet_speed/local_speed)**5)*(1-inlet_mach*angular_position)
    rotor_segment_noise = number_rotors/(2*np.pi)*quad(acoustic_integrand, 0.0, 2*np.pi, args=(airfoil_self_noise,sound_speed,inlet_speed))
    return rotor_segment_noise

def full_rotor_noise(hub_radius,rotor_radius,angular_speed,airfoil_self_noise,number_rotors,sound_speed,inlet_speed):
    full_rotor_noise = quad(rotor_segment_noise,hub_radius,rotor_radius,args=(airfoil_self_noise,angular_speed,number_rotors,airfoil_self_noise,sound_speed,inlet_speed))
    return full_rotor_noise


