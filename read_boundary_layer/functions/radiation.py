import numpy as np
from math import pi
import cmath
import matplotlib.pyplot as plt
import scipy.signal as sg
from scipy.special import fresnel
import numpy.lib.scimath as sci

from functions.Estar import Estar, fctf, fctg
from functions.ES import ES
from functions.E import E
from functions.turbulence import phi_vv

def L1_integral(omega,x,y,z,U,c_0,c,S_o): #x - streamwise z - spanwise y - wall normal
	k_bar = omega/c_0
	k_x = omega/U
	k_x_bar = omega/U*c/2
	k_z_bar = k_bar*z/S_o*c/2
	M = U/c_0
	beta = np.sqrt(1 - M**2)
	mu_bar = k_x_bar*M/beta**2
	kappa = sci.sqrt(mu_bar**2 - k_z_bar**2/beta**2)
	theta_1 = kappa - mu_bar*x/S_o
	theta_2 = mu_bar*(M - x/S_o) - np.pi/4

	integral = 1/np.pi*sci.sqrt(2/((k_x_bar + beta**2*kappa)*theta_1))*Estar(2*theta_1)*np.exp(1j*theta_2)
	
	return integral

def L2_integral(omega,x,y,z,U,c_0,c,S_o): #x - streamwise z - spanwise y - wall normal
	k_bar = omega/c_0
	k_x = omega/U
	k_x_bar = omega/U*c/2
	k_z_bar = k_bar*z/S_o*c/2
	M = U/c_0
	beta = np.sqrt(1 - M**2)
	mu_bar = k_x_bar*M/beta**2
	kappa = sci.sqrt(mu_bar**2 - k_z_bar**2/beta**2)
	theta_1 = kappa - mu_bar*x/S_o
	theta_2 = mu_bar*(M - x/S_o) - np.pi/4
	theta_3 = kappa + mu_bar*x/S_o

	integral = np.exp(1j*theta_2)/(np.pi*theta_1*np.sqrt(2*np.pi*sci.sqrt(2*np.pi*(k_x_bar+beta**2*kappa))))*\
	(1j*(1-np.exp(-2*1j*theta_1))+(1-1j)*(Estar(4*kappa)-sci.sqrt(2*kappa/theta_3)*np.exp(-2*1j*theta_1)*Estar(2*theta_3)))
	
	return integral
