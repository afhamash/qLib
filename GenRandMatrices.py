from qLinAlg import *

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
