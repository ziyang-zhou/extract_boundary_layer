import numpy as np
from scipy.special import factorial
from functions.Estar import Estar, fctf, fctg


def ES(arg):
    # create output variable
    Es = np.zeros(arg.shape, dtype=np.complex128)

    # compute indices for |arg|<2
    sa = np.abs(arg) < 2
    la = ~sa

    # series expansion for small arguments based on A&S p297, 7.1.5
    for j in range(21):
        Es[sa] += (-1)**j * (1j * arg[sa])**j / (factorial(j) * (2 * j + 1))

    # scaling factor
    Es[sa] = np.sqrt(2 / np.pi) * Es[sa]

    # use of Estar function for large arguments and division by sqrt(arg)
    Es[la] = Estar(arg[la]) / np.sqrt(arg[la])

    # regularisation for real negative and imag positive arg values
    k = la & ((arg.real < 0) & (arg.imag >= 0))
    Es[k] = -Es[k]

    return Es
