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

##### Distance measures #####

def fidelity(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Returns the (Uhlmann) fidelity between psd matrices P and Q

    Args:
        P (np.ndarray): PSD matrix argument for fidelity
        Q (np.ndarray): PSD matrix argument for fidelity

    Returns:
        float: The fidelity F(P,Q)
    """
    P_hf = MSR(P)
    return tr(MSR(P_hf @ Q @ P_hf))

def hilbert_schmidt_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Returns the HS (Euclidean) distance between matrices A and B.

    Args:
        A (np.ndarray): matrix argument
        B (np.ndarray): matrix argument

    Returns:
        float: the HS distance between A and B
    """
    Delta = A - B
    return np.sqrt(TIP(Delta, Delta))

def trace_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Returns the trace distance between matrices A and B.

    Args:
        A (np.ndarray): matrix argument
        B (np.ndarray): matrix argument

    Returns:
        float: the trace distance between A and B
    """
    Delta = A - B   # Compute the difference
    Delta_abs = MSR(Delta.conj().T@Delta)   # Compute the absolute value of the difference
    return tr(Delta_abs)    # return the trace of the Delta_abs as trace distance

def holevo_fidelity(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Returns the Holevo fidelity between psd matrices P and Q

    Args:
        P (np.ndarray): PSD matrix argument for fidelity
        Q (np.ndarray): PSD matrix argument for fidelity

    Returns:
        float: The Holevo fidelity F(P,Q)
    """
    return TIP(MSR(P), MSR(Q))

def matsumoto_fidelity(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Returns the Matsumoto fidelity between psd matrices P and Q

    Args:
        P (np.ndarray): PSD matrix argument for fidelity
        Q (np.ndarray): PSD matrix argument for fidelity

    Returns:
        float: The Matsumoto fidelity F(P,Q)
    """
    return tr(geometric_mean(P, Q))

def generalized_fidelity(P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.complex128:
    """
    Returns the generalized fidelity between P and Q at R

    Args:
        P (np.ndarray): PSD matrix argument
        Q (np.ndarray): PSD matrix argument
        R (np.ndarray): PD martrix base of generalized fidelity

    Returns:
        np.complex128: The generalized fidelity between P and Q at R
    """
    P_R = MSR(star_product(P,R))
    Q_R = MSR(star_product(Q,R))
    return tr(P_R @ mat_inv(R) @ Q_R)

