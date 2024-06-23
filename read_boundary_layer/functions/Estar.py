import numpy as np
from functions.phi_estimate import phi, Fn, Gn

def Estar(arg):
    # create output variable
    es = np.zeros_like(arg, dtype=np.complex)
    
    # check and compute indices of arg real and positive
    r = np.logical_and(np.imag(arg) == 0, np.real(arg) >= 0)
    c = np.logical_not(r)
    
    # use A&S 7.3.7-7.3.10 for real and positive arg
    f = fctf(np.sqrt(2 * arg[r] / np.pi))
    g = fctg(np.sqrt(2 * arg[r] / np.pi))
    
    C2 = 1/2 + f * np.sin(arg[r]) - g * np.cos(arg[r])
    S2 = 1/2 - f * np.cos(arg[r]) - g * np.sin(arg[r])
    
    es[r] = C2 - 1j * S2
    
    # use phi_0 for complex and real negative arguments
    if np.sum(c) != 0:
        # define real and imaginary part of arg
        es[c] = (1 - 1j) / 2 * phi(np.sqrt(1j * arg[c]))
    
    return es

# sub routines A&S p302, eq. 7.3.32
def fctf(x):
    f = (1 + 0.926 * x) / (2 + 1.792 * x + 3.104 * x**2)
    return f

# sub routines A&S p302, eq. 7.3.33
def fctg(x):
    g = 1 / (2 + 4.142 * x + 3.492 * x**2 + 6.670 * x**3)
    return g
