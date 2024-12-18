import numpy as np
import scipy as sp
import matplotlib as mpl

from numpy import linalg as la
from scipy import linalg as sla
from numpy.linalg import trace as tr
from numpy import sqrt as sr
from scipy.linalg import inv as inv
from scipy.linalg import pinv as pinv
from numpy.linalg import norm

##### Linear algebra #####

def is_herm(H: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Checks if a matrix H is Hermitian or not.

    Args:
        H (np.ndarray): Matrix whose Hermiticity is to be checked
        tol (float, optional): Error tolerence. Defaults to 1e-8.

    Returns:
        bool: Returns True if and only if H is Hermitian
    """
    return la.norm(H - H.conj().T) <= tol

def is_PSD(P: np.ndarray) -> bool:
    """
    Checks if a matrix P is positive semidefinite or not.

    Args:
        P (np.ndarray): Matrix whose positivity is to be checked

    Returns:
        bool: Returns True if and only if H is PSD
    """
    eig_vals = sla.eigvalsh(P)
    return is_herm(P) and eig_vals.min() >= 0


def is_density_matrix(D: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Checks if a matrix D is is a density matrix or not.

    Args:
        D (np.ndarray): Matrix that is to be checked

    Returns:
        bool: Returns True if and only if D is PSD
    """    
    return is_PSD(D) and np.abs(tr(D) - 1) <= 1e-8
    
def MSR(P: np.ndarray) -> np.ndarray:
    """
    Faster function to compute matrix square root of a 
    Hermitian PSD matrix.

    Args:
        P (np.ndarray): Hermitian PSD Matrix whose
                        square root is to be computed. 

    Returns:
        np.ndarray: a PSD matrix P_hf such that P_hf@P_hf = P
    """
    D, V = sp.linalg.eigh(P)
    return (V * np.sqrt(np.abs(D))) @ V.conj().T

def mat_inv(P: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute the inverse of a Hermitian matrix.

    Args:
        P (np.ndarray): A Hermitian matrix.

    Returns:
        np.ndarray: The inverse of the matrix.
    """
    D, V = sp.linalg.eigh(P)  # Eigendecomposition for Hermitian matrices
    D_inv = np.zeros_like(D)  # Initialize inverse eigenvalues with zeros
    non_zero_indices = np.abs(D) > tol  # Check which eigenvalues are nonzero
    D_inv[non_zero_indices] = 1 / D[non_zero_indices]  # Invert only nonzero eigenvalues
    return (V * D_inv) @ V.conj().T

def mat_power(H: np.ndarray, power: float) -> np.ndarray:
    """
    Compute the matrix power of a Hermitian matrix.

    Args:
        H (np.ndarray): A Hermitian matrix.
        power (float): The power to raise the matrix eigenvalues to.

    Returns:
        np.ndarray: The matrix raised to the specified power.
    """
    D, V = sp.linalg.eigh(H)
    return (V * D ** power) @ V.conj().T

def logM(P: np.ndarray, base: float = np.e) -> np.ndarray:
    """
    Compute the matrix logarithm (natural log) of a Hermitian matrix.

    Args:
        P (np.ndarray): A Hermitian matrix.

    Returns:
        np.ndarray: The matrix logarithm of the input matrix.
    
    Raises:
        ValueError: If eigenvalues are non-positive or base is invalid.
    """
    # Eigen decomposition
    D, V = sp.linalg.eigh(P)

    # Input validation: eigenvalues must be positive
    if np.any(D <= 0):
        raise ValueError("Matrix must have strictly positive eigenvalues to compute the logarithm.")

    # Compute logarithm for the given base
    log_D = np.log(D) / np.log(base)

    # Reconstruct the matrix logarithm
    return (V * log_D) @ V.conj().T

def expM(P: np.ndarray, base: float = np.e) -> np.ndarray:
    """
    Compute the matrix exponential of a Hermitian matrix for a specified base.

    Args:
        P (np.ndarray): A Hermitian matrix.
        base (float): The base of the exponential. Default is the natural base (e).

    Returns:
        np.ndarray: The matrix exponential of the input matrix for the given base.

    Raises:
        ValueError: If the base is not positive.
    """
    # Input validation: base must be positive
    if base <= 0:
        raise ValueError("Base must be positive for the matrix exponential.")

    # Eigen decomposition
    D, V = sp.linalg.eigh(P)

    # Compute the matrix exponential for the given base
    exp_D = np.exp(D * np.log(base))  # base^D = exp(D * log(base))

    # Reconstruct the matrix exponential
    return (V * exp_D) @ V.conj().T

def TIP(A: np.ndarray, B: np.ndarray) -> np.complex128:
    """
    Computes the trace inner product (HS/Frobenius/Euclidean) inner product
    between two matrices of same shape.

    Args:
        A (np.ndarray): A matrix 
        B (np.ndarray): Another matrix

    Returns:
        np.complex128: The dot-product between the two matrices
    
    Raises:
        ValueError: If the input matrices do not have the same shape.
    """
    if A.shape != B.shape:
        raise ValueError(f"Matrices must have the same shape, but got {A.shape} and {B.shape}")

    a = A.ravel()  # Flatten A into a 1D vector
    b = B.ravel()  # Flatten B into a 1D vector
    return np.vdot(a,b)

def star_product(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Returns the star-product of P with Q defined as Q^{1/2} @ P @ Q^{1/2}

    Args:
        P (np.ndarray): PSD matrix
        Q (np.ndarray): PSD matrix

    Returns:
        np.ndarray: returns the star-product P \star Q
    """
    Q_hf = MSR(Q)
    return Q_hf @ P @ Q_hf

def mat_division(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Returns the symmetrized matrix division frac{P}{Q} := Q^{-1/2} @ P @ Q^{-1/2}

    Args:
        P (np.ndarray): PSD matrix
        Q (np.ndarray): PSD matrix

    Returns:
        np.ndarray: returns the matrix division frac{P}{Q}
    """
    Q_inv = mat_inv(Q)
    return star_product(P, Q_inv) 

def geometric_mean(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Returns the geometric mean P#Q between positive definite P and Q

    Args:
        P (np.ndarray): PSD matrix
        Q (np.ndarray): PSD matrix

    Returns:
        np.ndarray: PSD matrix
    """
    return star_product(MSR(mat_division(P , Q)) , Q)
