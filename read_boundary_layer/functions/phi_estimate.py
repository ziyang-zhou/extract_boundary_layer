import numpy as np
from scipy.special import erf

def phi(arg):
    try:
        phi_0 = Faddeeva_erf(arg)
    except:
        # create output variable
        phi_0 = np.zeros_like(arg)
        
        # check and compute real and complex indices of arg
        r = np.imag(arg) == 0
        c = ~r
        
        # use built-in function of scipy for real arg
        phi_0[r] = erf(arg[r])
        
        # use A&S p299, eq. 7.1.29 for complex arguments
        if np.sum(c) != 0:
            # define real and imaginary part of arg
            x = np.real(arg[c])
            y = np.imag(arg[c])
            dxy = 2 * x * y
            
            for n in range(1, 11):
                phi_0[c] = phi_0[c] + np.exp(-n**2/4) / (n**2 + 4 * x**2) * (Fn(x, y, dxy, n) + 1j * Gn(x, y, dxy, n))
            
            # dummy variable
            t2 = np.exp(-x**2) / (2 * np.pi * x) * ((1 - np.cos(dxy)) + 1j * np.sin(dxy))
            
            # singularity if x = 0
            t2[x == 0] = 1j * y[x == 0] / np.pi
            
            phi_0[c] = phi_0[c] * (2 / np.pi) * np.exp(-x**2) + erf(x) + t2
        
    return phi_0

# sub routines A&S p299, eq. 7.1.29

def Fn(x, y, dxy, n):
    fn = 2 * x - 2 * x * np.cosh(n * y) * np.cos(dxy) + n * np.sinh(n * y) * np.sin(dxy)
    return fn

def Gn(x, y, dxy, n):
    gn = 2 * x * np.cosh(n * y) * np.sin(dxy) + n * np.sinh(n * y) * np.cos(dxy)
    return gn
