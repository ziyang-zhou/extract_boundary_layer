import matplotlib as mlib
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import special
import pylab as plb

#x is streamwise, y is wall normal and z is spanwise 0.0,1.0,0.0
def phi_vv(x,y,z,U_0,ti,L,c_0,f):
	omega = 2*np.pi*f
	M=U_0/c_0 #mach number
	k=omega/c_0
	beta=(1-M**2)**0.5
	u_prime_sq = (ti*U_0)**2
	sigma=(x**2+(beta**2)*(z**2 + y**2))**0.5 #modified observer distance
	ke=(np.sqrt(np.pi)/L)*(special.gamma(5/6)/special.gamma(1/3)) #typo found in original paper
	ke_const = (np.sqrt(np.pi))*(special.gamma(5/6)/special.gamma(1/3))
	kx=(omega/U_0)/ke
	kz=(k*z/sigma)/ke
	phi = 4/(9*np.pi)*(u_prime_sq/(ke**2))*(kx**2+kz**2)/((1+kx**2+kz**2)**(7/3)) # Von Karman
	#phi = (91/(36*np.pi))*(u_prime_sq/(ke**2))*((kx**2+kz**2)/((1+kx**2+kz**2)**(19/6))) # RDT
	return phi
