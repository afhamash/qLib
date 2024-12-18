from mybase import *

##### Linear algebra #####

def H(A: np.ndarray) -> np.ndarray:
    """
    Returns the Adjoint (conjugate transpose) of a matrix A

    Args:
        A (np.ndarray): Input matrix

    Returns:
        np.ndarray: Output matrix  A^*
    """
    return A.conj().T

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
    return is_herm(P) and eig_vals.min() >= -1e-10

def is_density_matrix(D: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Checks if a matrix D is is a density matrix or not.

    Args:
        D (np.ndarray): Matrix that is to be checked

    Returns:
        bool: Returns True if and only if D is PSD
    """    
    return is_PSD(D) and np.abs(tr(D) - 1) <= 1e-8

def is_unitary(U: np.ndarray) -> bool:
    """
    Checks if the input matrix is unitary 
    Args:
        U (np.ndarray): Input matrix

    Returns:
        bool: True if and only if U is unitary
    """
    d = U.shape[0]
    Id = eye(d)
    return np.allclose(Id, U@H(U)) and np.allclose(Id, H(U)@U)

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

def multi_kron(*matrices: np.ndarray) -> np.ndarray:
    """
    Kronocker product of the list of input matrices.

    Args:
        matrices: An ordered list of np.ndarrays whose kronecker product is to be taken.
    Returns:
        np.ndarray: The kronecker product of all the matrices
    """
    A = 1 + 0j
    for B in matrices:
        A = np.kron(A, B) 
    return A

def partial_trace(P: np.ndarray, dims: list, TracedOutSubsystem: int) -> np.ndarray:
  """_summary_

  Args:
      P (np.ndarray): Bipartite matrix of shape (d0*d1, d0*d1) to be partial traced
      dims (list): = [d0, d1]. Current implementation holds only for bipartite systems.
      TracedOutSubsystem (int): The subsystem being traced out

  Raises:
      Exception: _description_

  Returns:
      np.ndarray: _description_
  """
  if len(dims) != 2: raise ValueError('Number of systems (and thereby dimensions) must be exactly 2.')
  
  d0, d1 = dims
  P_tensor = P.reshape(d0, d1, d0, d1)
  T = TracedOutSubsystem
  
  if T in [0, 1]:
    P_tensor = np.trace(P_tensor, axis1 = T, axis2 = T+2)
    return P_tensor
  else:
    raise ValueError('TracedOutSubsystem must be either 0 or 1')
  
def choi_projection(P: np.ndarray, dX: int, dY: int) -> np.ndarray:
    """
    Computes the Choi projection C of a bipartite matrix P.
    P in Pos(X, Y) will be projected to Choi(X, Y),
    which is a bipartite psd matrix with X-marginal equal to Identity.  

    Args:
        P (np.ndarray): Must be a PSD matrix of shape (dX * dY, dX * dY).
        dX (int): Dimension of X subspace
        dY (int): Dimension of Y subspace

    Returns:
        np.ndarray: The Bures projection to the set of Choi matrices
    """
    if not is_PSD(P): 
        raise ValueError('Input matrix must be positive semidefinite.')

    d = P.shape[0]
    if d != dX*dY:
        raise ValueError('Dimension mismatch. dX*dY must equal dimension of Input Matrix.')

    PX = partial_trace(P, [dX, dY], 1)      # Compute the X-marginal of P

    # Compute the projection to Choi states
    C = mat_division(P, np.kron(PX, np.eye(dY)))        # Inefficient implementation for large matrices.
    return C

def is_choi(P: np.ndarray, dX: int, dY: int) -> bool:
    """
    Checks if P in Choi(X, Y)

    Args:
        P (np.ndarray): Matrix whose Choi-ness is to be checked
        dX (int): dimension of X (input) space
        dY (int): dimension of Y (output) space

    Returns:
        bool: True if and only if P in Choi(X, Y)
    """
    if not is_PSD(P):
        print('Input matrix is not positive semidefinite')
        return False

    PX = partial_trace(P, [dX, dY], 1)      # Compute the X-marginal of P
    if norm(PX - np.eye(dX)) > 1e-6:
        print('X-marginal of Input matrix is not Identity')
        return False

    return True

def matrix_modulus(A: np.ndarray) -> np.ndarray:

    """
    Returns the absolute value |A| = sqrt(A^* A) of a matrix A.
    Args:
        A (np.ndarray): Arbitrary matrix

    Returns:
        np.ndarray: |A| = sqrt(A^* A).
    """
    return MSR(A.conj().T @ A)
