import numpy as np
import scipy as sp
import matplotlib as mpl

from qLinAlg import *

from numpy import linalg as la
from scipy import linalg as sla
from numpy.linalg import trace as tr
from numpy import sqrt as sr
from scipy.linalg import inv as inv
from scipy.linalg import pinv as pinv
from numpy.linalg import norm


##### Random matrix generators #####

def PSD_matrix(d: int, r: int, seed: int = None) -> np.ndarray:
    """
    Generates a random d-dimensional r-rank positive semidefinite matrix

    Args:
        d (int) : Dimension of the density matrix.
        r (int) : Rank of the density matrix.
        seed (int, optional) : Seed for RNG repeoducability. 

    Returns:
        np.ndarray : A density matrix of shape (d, d) with rank `r`.
    """
    if seed is not None:
        np.random.seed(seed)
        
    X = np.random.rand(d, r) + 1j*np.random.rand(d, r)
    P = X @ X.conj().T    # Construct random PSD matrix
    return P

def density_matrix(d: int, r:int, seed: int = None) -> np.ndarray:
    """
    Generates a random d-dimensional r-rank density matrix.

    Args:
        d (int) : Dimension of the density matrix.
        r (int) : Rank of the density matrix.
        seed (int, optional) : Seed for RNG repeoducability. 

    Returns:
        np.ndarray : A density matrix of shape (d, d) with rank `r`.
    """
    if seed is not None:
        np.random.seed(seed)

    P = PSD_matrix(d, r)    # Generate random PSD matrix
    rho = P / tr(P)       # Normalize to obtain density matrix
    return rho

# def unitary_matrix(d: int, seed: int = None) -> np.ndarray:
