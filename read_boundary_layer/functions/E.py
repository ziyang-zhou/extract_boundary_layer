import numpy as np
from scipy.special import factorial
from functions.Estar import Estar, fctf, fctg

def E(arg):
    # create output variable
    Es = np.zeros_like(arg, dtype=complex)
    
    # check and compute indices of arg real and positive
    r = np.imag(arg) == 0 and np.real(arg) >= 0
    c = ~r
    
    Es[r] = -Estar(-arg[r]) / 1j
    
    # use phi_0 for complex and real negative arguments
    if np.sum(c) != 0:
        # define real and imaginary part of arg
        Es[c] = (1 + 1j) / 2 * Phi_0(np.sqrt(-1j * arg[c]))
    
    return Es
