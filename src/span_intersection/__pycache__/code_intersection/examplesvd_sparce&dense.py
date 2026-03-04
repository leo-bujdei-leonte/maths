"""
Null Space Computation: Three Methods Compared
==============================================
1. Dense SVD (scipy.linalg.svd) - Complete decomposition
2. Built-in null_space() (scipy.linalg.null_space) - Direct method
3. Sparse SVD (scipy.sparse.linalg.svds) - For large sparse matrices
"""

import numpy as np
from scipy.linalg import svd as dense_svd, null_space
from scipy.sparse.linalg import svds, splu
from scipy.sparse import csr_matrix, csc_array
import scipy.linalg.interpolative as sli
import scipy as sp



# METHOD 1: DENSE SVD 

def null_space_dense_svd(A, tol=1e-5):
    """
    Find null space using dense SVD (full decomposition).
    
    Args:
        A: m x n matrix (numpy array)
        tol: tolerance for determining singular values near zero
    
    Returns:
        null_basis: n x k matrix where columns form null space basis
        null_indices: indices of singular vectors corresponding to null space
        singular_values: all singular values
    """
    U, s, Vt = dense_svd(A, full_matrices=True)
    
    # Find singular values below tolerance (these correspond to null space)
    null_mask = s < tol
    print(f"Rank = {np.sum(s>tol)}")
    null_indices = np.where(null_mask)[0]
    
    # Null space vectors are RIGHT singular vectors (rows of Vt)
    # corresponding to zero/near-zero singular values
    null_basis = Vt[null_indices, :].T  # Transpose to get column vectors
    
    return null_basis, null_indices, s


# METHOD 2: SCIPY BUILT-IN null_space()

def null_space_builtin(A):
    """
    Find null space using scipy's built-in null_space function.
    This is the simplest and most reliable method for dense matrices.
    
    Args:
        A: m x n matrix (numpy array)
    
    Returns:
        null_basis: n x k matrix where columns form null space basis
    """
    return null_space(A)


# METHOD 3: SPARSE SVD 

def null_space_sparse_svd(A_sparse, k_null=None, tol=1e-10):
    """
    Find null space using sparse SVD (Lanczos-based).
    Efficient for large sparse matrices.
    
    Args:
        A_sparse: m x n sparse matrix (scipy.sparse format)
        k_null: number of null space dimensions to compute
        tol: tolerance for determining singular values near zero
    
    Returns:
        null_basis: n x k matrix where columns form null space basis
        singular_values: computed smallest singular values
    """
    m, n = A_sparse.shape
    
    if k_null is None:
        # Compute several small singular values
        k_compute = min(n - 1, 10)
    else:
        k_compute = min(k_null, n - 1)
    
    try:
        # Compute smallest singular values
        U, s, Vt = svds(A_sparse, k=k_compute, which='SM')
        
        # Find singular values below tolerance
        null_mask = s < tol
        null_basis = Vt[null_mask, :].T
        
        return null_basis, s
    
    except Exception as e:
        print(f"Sparse SVD failed: {e}")
        return None, None


# METHOD 4: LU NULLSPACE 

def null_lu(A):
    """
    Find null space using sparse LU.
    
    
    Args:
        A_sparse: m x n  matrix 
    
    Returns:
        
    """
    P, L, U = sp.linalg.lu(A)

    print(f"The nullspace can be calculated from {U}.")

    
    return U

# Source - https://stackoverflow.com/a
# Retrieved 2026-01-28, License - CC BY-SA 3.0

def squarify(M,val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)


# METHOD 5: LU SPARSE NULLSPACE

def null_lusp(A):
    """
    Find null space using sparse LU.
    
    
    Args:
        A_sparse: m x n  matrix 
    
    Returns:
    """
    y=np.zeros(A.shape[1])
    
    A_padded=squarify(A,0)
    csc_format=csc_array(A_padded)
    inv_A=splu(csc_format)

    u=inv_A.U
    x=inv_A.solve(y)
    return x


# VERIFICATION FUNCTION

def verify_null_space(A, null_basis, method_name):
    """Verify that computed null space vectors satisfy A @ v = 0"""
    print(f"\n{'-' * 70}")
    print(f"VERIFICATION: {method_name}")
    print(f"{'-' * 70}")
    print(f"Null space dimension: {null_basis.shape[1]}")
    print()
    
    for i in range(null_basis.shape[1]):
        v = null_basis[:, i]
        residual = np.linalg.norm(A @ v)
        print(f"  Vector {i}: ||A @ v_{i}|| = {residual:.2e}")
    print()
    
    # Check entire null space basis at once
    residuals = np.linalg.norm(A @ null_basis, axis=0)
    max_residual = np.max(residuals)
    print(f"Maximum residual: {max_residual:.2e}")
    print(f"All vectors in null space" if max_residual < 1e-8 else "Issues detected")


# EXAMPLE: Rank-Deficient Matrix

if __name__ == "__main__":
    # Create a rank-deficient matrix (rank 2, dimension 5)
    A = np.array([
        [1.0, 2.0, 3.0, 4.0, 0.0],
        [2.0, 4.0, 6.0, 8.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.9],
        [3.0, 0.0, 3.0, 0.0, 0.0],
    ], dtype=float)
    
    print("=" * 70)
    print("MATRIX INFORMATION")
    print("=" * 70)
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix A:")
    print(A)
    print()
    print(f"Numpy Rank: {np.linalg.matrix_rank(A)}")
    print(f"Expected null space dimension: {A.shape[1] - np.linalg.matrix_rank(A)}")
    print()
    print(f"Check if the matrix is padded: {squarify(A,0)}")
    


    # METHOD 1: Dense SVD
    print("=" * 70)
    print("METHOD 1: Dense SVD")
    print("=" * 70)
    null_1, null_indices, s = null_space_dense_svd(A)
    print(f"Singular values: {s}")
    print(f"Null space basis shape: {null_1.shape}")
    print("Null space basis vectors:")
    print(null_1)
    verify_null_space(A, null_1, "Dense SVD")
    
    # METHOD 2: Built-in null_space()
    
    print("=" * 70)
    print("METHOD 2: scipy.linalg.null_space()")
    print("=" * 70)
    null_2 = null_space_builtin(A)
    print(f"Null space basis shape: {null_2.shape}")
    print("Null space basis vectors:")
    print(null_2)
    verify_null_space(A, null_2, "Built-in null_space()")
    

    # METHOD 3: Sparse SVD
    print("=" * 70)
    print("METHOD 3: Sparse SVD")
    print("=" * 70)
    A_sparse = csr_matrix(squarify(A,0))
    null_3, sparse_s = null_space_sparse_svd(A_sparse, k_null=3)
    
    if null_3 is not None:
        print(f"Smallest singular values computed: {sparse_s}")
        print(f"Null space basis shape: {null_3.shape}")
        print("Null space basis vectors:")
        print(null_3)
        verify_null_space(A, null_3, "Sparse SVD")
    else:
        print("Sparse SVD computation failed")
    

    # METHOD 4: LU decomposition
    print("=" * 70)
    print("METHOD 4: LU decomposition")
    print("=" * 70)
    null_4 = sp.linalg.null_space(null_lu(A))
    rank=np.sum(np.diagonal(null_lu(A))>0)
    
    if null_4 is not None:
        print(f"Null space basis shape: {null_4.shape}")
        print("Null space basis vectors:")
        print(f"Rank obtained with LU decomposition is {rank}")
        verify_null_space(A, null_4, "LU decomposition")
    else:
        print("LU decomposition failed computation failed")
    
# METHOD 5: sparse LU decomposition
    print("=" * 70)
    print("METHOD 5: Sparse LU decomposition")
    print("=" * 70)
    #print(f"The null_space is: {null_lusp(A)}")
    lu=splu(csc_array(squarify(A,0)))
    print(f"Try rank: {sum(abs(np.diag(lu.U.toarray())) > 1e-10)}")
    """
    if null_4 is not None:
        print(f"Null space basis shape: {null_4.shape}")
        print("Null space basis vectors:")
        print(f"Rank obtained with LU decomposition is {rank}")
        verify_null_space(A, null_4, "LU decomposition")
    else:
        print("LU decomposition failed computation failed")
    """

    # COMPARISON
    print("=" * 70)
    print("COMPARISON: All Methods")
    print("=" * 70)
    print(f"Method 1 (Dense SVD):       null space dim = {null_1.shape[1]}")
    print(f"Method 2 (Built-in):        null space dim = {null_2.shape[1]}")
    if null_3 is not None:
        print(f"Method 3 (Sparse SVD):      null space dim = {null_3.shape[1]}")
    if null_4 is not None:
        print(f"Method 4 (LU decomposition): null space dim = {null_4.shape[1]}")
    print()
   
