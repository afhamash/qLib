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

def unitary_matrix(d: int) -> np.ndarray:
    """
    Returns an d \times d Unitary matrix 

    Args:
        d (int): dimension of unitary matrix

    Returns:
        np.ndarray: Unitary matrix
    """
    A = np.random.rand(d,d)
    return sla.polar(A)[0]

def prob_vec(n: int) -> np.ndarray:
    """
    Return an n-dimensional proability vector

    Args:
        n (int): dimension of probability vector

    Returns:
        np.ndarray: an n-dimensional probability vector
    """
    p = np.random.rand(n)
    p /= np.sum(p)      # Normalize to sum to 1
    return p 

def choi_State(dX: int, dY: int, r: int) -> np.ndarray:
    """
    Generates a random Choi state by Bures-projecting a random 
    bipartite q

    Args:
        dX (int): dimension of X (input) subspace 
        dY (int): dimension of Y (output) subspace
        r (int): Rank of Choi matrix (which is equal to the Choi rank of the channel) 

    Returns:
        np.ndarray: A random Choi matrix
    """
    P = PSD_matrix(dX*dY, r)
    return choi_projection(P, dX, dY)