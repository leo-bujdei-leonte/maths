"""
Optimized non-commutative algebra implementation for computing intersections
of U(W_+) subspaces with non-multiplication rules.

- Polynomial representation with operator overloading
- Rank computation using NumPy and LinBox backends

Multiplication Rule:
    e_n * e_m = e_m * e_n + (m-n) * e_{n+m} when n > m
"""

from __future__ import annotations
from typing import Tuple, Dict, Iterator, List, Optional, Union
from functools import lru_cache, reduce
from dataclasses import dataclass, field
from math import gcd
from fractions import Fraction
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds
from collections import defaultdict
from tqdm import tqdm
import warnings
import scipy

Monomial = Tuple[int, ...]  # Ordered tuple of indices representing e_{i1} * e_{i2} * ...
CoeffDict = Dict[Monomial, int]  # Sparse representation: monomial -> coefficient


#RANK COMPUTATION SYSTEM

class RankComputer:
    """
    Compute matrix rank using different backends.
    """
    
    def __init__(self, backend: str = "numpy"):
        """
        Initialize rank computer.
        
        Args:
            backend: Either "numpy" or "linbox"
        """
        self.backend = backend.lower()
        
        if self.backend == "linbox":
            try:
                import linbox
                self.linbox = linbox
            except ImportError:
                raise ImportError(
                    "LinBox not installed. Install with: pip install linbox "
                    "or use backend='numpy' instead."
                )
        elif self.backend != "numpy":
            raise ValueError(f"Unknown backend: {backend}. Choose 'numpy' or 'linbox'.")
    
    def rank_numpy(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        tol: Optional[float] = None,
        verbose: bool = False
    ) -> Tuple[int, Dict[str, any]]:
        """
        Compute rank using NumPy's SVD-based method (np.linalg.matrix_rank).
        
        Works with dense and sparse matrices.
        
        Args:
            matrix: Input matrix (dense array or sparse CSR)
            tol: Tolerance for singular values. If None, uses NumPy's default.
            verbose: Print debug information
            
        Returns:
            Tuple of (rank, debug_info_dict)
        """
        # Convert sparse to dense if needed
        if isinstance(matrix, csr_matrix):
            if verbose:
                print(f"  Converting sparse matrix to dense ({matrix.shape[0]} × {matrix.shape[1]})")
            matrix_dense = matrix.toarray()
        else:
            matrix_dense = np.asarray(matrix)
        
        if verbose:
            print(f"  Matrix shape: {matrix_dense.shape}")
            print(f"  Data type: {matrix_dense.dtype}")
        
        # Compute rank
        if tol is not None:
            rank = np.linalg.matrix_rank(matrix_dense, tol=tol)
            tol_used = tol
        else:
            rank = np.linalg.matrix_rank(matrix_dense)
            # Estimate the tolerance used by NumPy
            tol_used = max(matrix_dense.shape) * np.finfo(float).eps * np.max(np.abs(matrix_dense))
        """
        # Compute SVD for additional info
        U, s, Vt = np.linalg.svd(matrix_dense, full_matrices=False)
        
        # Compute condition number
        if len(s) > 0 and s[-1] != 0:
            condition_number = s[0] / s[-1]
        else:
            condition_number = np.inf
        """
        debug_info = {
            "method": "numpy_svd",
            "rank": rank,
            "shape": matrix_dense.shape,
            "tolerance": tol_used
        }
        
        if verbose:
            print(f"  Rank: {rank}")
            print(f"  Tolerance used: {tol_used:.2e}")
        
        return rank, debug_info
    
    def rank_linbox(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        method: str = "adaptive",
        verbose: bool = False
    ) -> Tuple[int, Dict[str, any]]:
        """
        Compute rank using LinBox (exact computation over rationals).
        
        LinBox can compute exact rank over the rationals, which is useful
        for systems with integer or rational coefficients.
        
        Args:
            matrix: Input matrix (must be dense or converted)
            method: "gauss", "adaptive", or "blackbox"
            verbose: Print debug information
            
        Returns:
            Tuple of (rank, debug_info_dict)
        """
        # Convert to dense if needed
        if isinstance(matrix, csr_matrix):
            if verbose:
                print(f"  Converting sparse matrix to dense ({matrix.shape[0]} × {matrix.shape[1]})")
            matrix_dense = matrix.toarray()
        else:
            matrix_dense = np.asarray(matrix)
        
        if verbose:
            print(f"  Matrix shape: {matrix_dense.shape}")
            print(f"  Using LinBox method: {method}")
        
        try:
            # Create LinBox field (rationals)
            F = self.linbox.RationalField()
            
            # Create LinBox matrix
            m, n = matrix_dense.shape
            A = self.linbox.DenseMatrix(F, m, n)
            
            # Populate matrix
            for i in range(m):
                for j in range(n):
                    val = matrix_dense[i, j]
                    # Convert to integer or rational
                    if val != 0:
                        A[i, j] = int(val) if val == int(val) else val
            
            # Compute rank using specified method
            rank = self.linbox.rank(A)
            
            debug_info = {
                "method": f"linbox_{method}",
                "rank": rank,
                "shape": matrix_dense.shape,
                "backend": "LinBox (exact)",
                "field": "Rationals"
            }
            
            if verbose:
                print(f"  Rank: {rank}")
                print(f"  Computed exactly over rationals")
            
            return rank, debug_info
            
        except Exception as e:
            print(f"  [ERROR] LinBox computation failed: {e}")
            raise
    
    def compute_rank(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        **kwargs
    ) -> Tuple[int, Dict[str, any]]:
        """
        Compute rank using the configured backend.
        
        Args:
            matrix: Input matrix
            **kwargs: Backend-specific arguments
            
        Returns:
            Tuple of (rank, debug_info)
        """
        if self.backend == "numpy":
            return self.rank_numpy(matrix, **kwargs)
        elif self.backend == "linbox":
            return self.rank_linbox(matrix, **kwargs)


#MONOMIAL REDUCTION

@lru_cache(maxsize=10000)
def reduce_monomial(seq: Monomial) -> Tuple[Tuple[Monomial, int], ...]:
    """
    Reduce a monomial sequence to sorted form using the non-commutative rule:
        e_n * e_m = e_m * e_n + (m-n) * e_{n+m} when n > m
    
    Results are cached indefinitely for maximum speed on repeated computations.
    
    Args:
        seq: Tuple of indices representing a monomial product
        
    Returns:
        Tuple of (monomial, coefficient) pairs representing the reduced form
        
    Example:
        reduce_monomial((3, 1)) -> (((1, 3), 1), ((4,), -2))
        which represents: e_3 * e_1 = e_1 * e_3 - 2*e_4
    """
    # Base cases: empty or single element is already reduced
    if len(seq) <= 1:
        return ((seq, 1),) if seq else (((), 1),)
    
    # Check if already in increasing order
    if all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)):
        return ((seq, 1),)
    
    # Find first out-of-order pair where n > m
    for i in range(len(seq) - 1):
        n, m = seq[i], seq[i + 1]
        if n > m:
            left = seq[:i]
            right = seq[i + 2:]
            
            # Term 1: swap the pair (e_m * e_n)
            swapped = left + (m, n) + right
            term1_pairs = reduce_monomial(swapped)
            
            # Term 2: correction term (m - n) * e_{n+m}
            correction_coeff = m - n
            corrected = left + (n + m,) + right
            term2_pairs = reduce_monomial(corrected)
            
            # Combine results efficiently
            result: Dict[Monomial, int] = {}
            for mono, coeff in term1_pairs:
                result[mono] = result.get(mono, 0) + coeff
            for mono, coeff in term2_pairs:
                result[mono] = result.get(mono, 0) + correction_coeff * coeff
            
            # Filter zeros and return as tuple for caching
            return tuple((m, c) for m, c in result.items() if c != 0)
    
    # Should never reach here
    return ((seq, 1),)


def reduce_to_dict(seq: Monomial) -> CoeffDict:
    """Convert cached tuple result to dictionary for easier manipulation."""
    return {mono: coeff for mono, coeff in reduce_monomial(seq)}


#POLYNOMIAL CLASS

@dataclass
class Polynomial:
    """
    Represents a polynomial as a sparse linear combination of monomials.
    
    Internal representation: Dict[Monomial, int] where:
        - Keys are tuples of indices (monomials)
        - Values are integer coefficients
        - Empty tuple () represents the constant 1
    
    Supports standard arithmetic operations: +, -, *, **, scalar multiplication
    """
    
    terms: CoeffDict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure all terms have non-zero coefficients."""
        self.terms = {m: c for m, c in self.terms.items() if c != 0}
    
    # Factory Methods
    
    @classmethod
    def zero(cls) -> Polynomial:
        """Create the zero polynomial."""
        return cls({})
    
    @classmethod
    def constant(cls, value: int) -> Polynomial:
        """Create a constant polynomial."""
        return cls({(): value}) if value != 0 else cls.zero()
    
    @classmethod
    def from_monomial(cls, monomial: Monomial, coeff: int = 1) -> Polynomial:
        """Create a polynomial from a single monomial."""
        if coeff == 0:
            return cls.zero()
        return cls({tuple(monomial): coeff})
    
    @classmethod
    def basis_element(cls, index: int) -> Polynomial:
        """Create e_i for a given index i."""
        return cls.from_monomial((index,), 1)
    
    # Arithmetic Operations
    
    def __add__(self, other: Polynomial) -> Polynomial:
        """Add two polynomials."""
        result = dict(self.terms)
        for mono, coeff in other.terms.items():
            result[mono] = result.get(mono, 0) + coeff
        return Polynomial(result)
    
    def __radd__(self, other):
        """Right addition (for sum() support)."""
        if other == 0:
            return self
        return self.__add__(other)
    
    def __neg__(self) -> Polynomial:
        """Negate all coefficients."""
        return Polynomial({m: -c for m, c in self.terms.items()})
    
    def __sub__(self, other: Polynomial) -> Polynomial:
        """Subtract two polynomials."""
        return self + (-other)
    
    def __mul__(self, other: Union[Polynomial, int]) -> Polynomial:
        """
        Multiply by another polynomial or scalar.
        Uses cached monomial reductions for efficiency.
        """
        # Scalar multiplication
        if isinstance(other, int):
            if other == 0:
                return Polynomial.zero()
            return Polynomial({m: c * other for m, c in self.terms.items()})
        
        # Polynomial multiplication
        if isinstance(other, Polynomial):
            if not self.terms or not other.terms:
                return Polynomial.zero()
            
            result: Dict[Monomial, int] = defaultdict(int)
            
            for m1, c1 in self.terms.items():
                for m2, c2 in other.terms.items():
                    # Concatenate monomials and reduce
                    concat = m1 + m2
                    reduced = reduce_to_dict(concat)
                    
                    # Accumulate terms
                    for mono, mono_coeff in reduced.items():
                        result[mono] += c1 * c2 * mono_coeff
            
            return Polynomial(dict(result))
        
        return NotImplemented
    
    def __rmul__(self, other):
        """Right multiplication (for scalar * polynomial)."""
        if isinstance(other, int):
            return self.__mul__(other)
        return NotImplemented
    
    def __pow__(self, exponent: int) -> Polynomial:
        """
        Raise polynomial to integer power using binary exponentiation.
        Time complexity: O(log exponent)
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Only non-negative integer powers supported")
        
        if exponent == 0:
            return Polynomial.constant(1)
        if exponent == 1:
            return Polynomial(dict(self.terms))
        
        # Binary exponentiation
        result = Polynomial.constant(1)
        base = Polynomial(dict(self.terms))
        exp = exponent
        
        while exp > 0:
            if exp & 1:  # If exp is odd
                result = result * base
            base = base * base
            exp >>= 1
        
        return result
    
    # Properties
    
    def is_zero(self) -> bool:
        """Check if polynomial is zero."""
        return len(self.terms) == 0
    
    def degree(self) -> int:
        """Return the maximum sum of indices in any monomial."""
        if not self.terms:
            return -1  # Degree of zero polynomial
        return max(sum(mono) for mono in self.terms.keys())
    
    def num_terms(self) -> int:
        """Return number of non-zero terms."""
        return len(self.terms)
    
    # String Representation
    
    def __repr__(self) -> str:
        """Rep for debugging."""
        if not self.terms:
            return "Polynomial({})"
        return f"Polynomial({dict(self.terms)})"
    
    def __str__(self) -> str:
        """Readable string representation."""
        if not self.terms:
            return "0"
        
        # Sort terms for consistent output: by degree, then lexicographically
        sorted_items = sorted(
            self.terms.items(),
            key=lambda kv: (len(kv[0]), kv[0])
        )
        
        parts = []
        for mono, coeff in sorted_items:
            if mono == ():  # Constant term
                parts.append(str(coeff))
            else:
                mono_str = "*".join(f"e_{i}" for i in mono)
                if coeff == 1:
                    parts.append(mono_str)
                elif coeff == -1:
                    parts.append(f"-{mono_str}")
                else:
                    parts.append(f"{coeff}*{mono_str}")
        
        # Join with proper signs
        result = parts[0]
        for part in parts[1:]:
            if part.startswith("-"):
                result += f" - {part[1:]}"
            else:
                result += f" + {part}"
        
        return result
    
    def __bool__(self):
        """Boolean conversion: False if zero, True otherwise."""
        return not self.is_zero()
    
    # Methods
    
    def copy(self) -> Polynomial:
        """Create a copy of the polynomial."""
        return Polynomial(dict(self.terms))
    
    def expand(self) -> Polynomial:
        """Return expanded form (already expanded in our representation)."""
        return self.copy()


#CONVENIENCE FUNCTIONS

def e(index: int) -> Polynomial:
    """
    Create basis element e_i.
    
    Args:
        index: The subscript i
        
    Returns:
        Polynomial representing e_i
    """
    return Polynomial.basis_element(index)


def g(n: int, m: int) -> Polynomial:
    """
    Construct the special element g(n,m) used in intersection computations.
    
    Formula: g(n,m) = e_n * e_m - e_1 * e_{n+m-1} + (n-1) * e_{n+m}
    
    Args:
        n: First index
        m: Second index
        
    Returns:
        Polynomial representing g(n,m)
    """
    term1 = e(n) * e(m)
    term2 = e(1) * e(n + m - 1)
    term3 = (n - 1) * e(n + m)
    return term1 - term2 + term3


#PARTITION GENERATION

def integer_partitions(n: int, min_part: int = 1) -> Iterator[Tuple[int, ...]]:
    """
    Generate all integer partitions of n with parts in increasing order.
    
    Args:
        n: Integer to partition
        min_part: Minimum part size (used in recursion)
        
    Yields:
        Tuples representing partitions in increasing order
        
    Example:
        list(integer_partitions(4)) gives:
        [(4,), (1,3), (2,2), (1,1,2), (1,1,1,1)]
    """
    yield (n,)
    for i in range(min_part, n // 2 + 1):
        for partition in integer_partitions(n - i, i):
            yield (i,) + partition


def generate_uw_basis(n: int, verbose: bool = False) -> List[Polynomial]:
    """
    Generate the Poincaré-Birkhoff-Witt (PBW) basis for U(W_+)_n.
    
    The basis consists of all products e_{i1} * e_{i2} * ... * e_{ik}
    where i1 + i2 + ... + ik = n and the product is reduced to normal form.
    
    Args:
        n: The degree/weight
        verbose: If True, print progress information
        
    Returns:
        List of basis elements as Polynomial objects
    """
    if verbose:
        print(f"Generating UW_basis({n})...")
    
    partitions = list(integer_partitions(n))
    basis = []
    
    # Reverse order to match original behavior
    for partition in reversed(partitions):
        # Build product e_{i1} * e_{i2} * ... * e_{ik}
        product = Polynomial.constant(1)
        for index in partition:
            product = product * e(index)
        basis.append(product)
    
    if verbose:
        print(f"[GEN] Finished generate_uw_basis({n})", flush=True)
    
    return basis


#VECTOR RATIONALIZATION

def rationalize_vector(vec: np.ndarray, max_denominator: int = 100) -> List[int]:
    """
    Convert a floating-point vector to integer coefficients.
    
    Process:
    1. Convert each entry to a rational approximation
    2. Find LCM of denominators
    3. Scale to integers
    4. Reduce by GCD
    
    Args:
        vec: Input vector with floating-point values
        max_denominator: Maximum denominator for rational approximation
        
    Returns:
        List of integer coefficients
    """
    # Convert to fractions
    fractions = []
    for value in vec:
        if abs(value) < 1e-10:
            fractions.append(Fraction(0))
        else:
            fractions.append(Fraction(value).limit_denominator(max_denominator))
    
    # Extract non-zero denominators
    denominators = [f.denominator for f in fractions if f != 0]
    if not denominators:
        return [0] * len(vec)
    
    # Compute LCM of denominators
    lcm = reduce(lambda a, b: a * b // gcd(a, b), denominators, 1)
    
    # Scale to integers
    int_vec = [int(f * lcm) for f in fractions]
    
    # Reduce by GCD
    vec_gcd = reduce(gcd, (abs(x) for x in int_vec if x != 0), 0)
    if vec_gcd > 0:
        int_vec = [x // vec_gcd for x in int_vec]
    return int_vec


def fully_expand(poly: Polynomial) -> Polynomial:
    """Fully expand all monomials in a polynomial to normal form."""
    result: Dict[Monomial, int] = defaultdict(int)

    for mono, coeff in poly.terms.items():
        # reduce_monomial returns a tuple of (monomial, coeff)
        reduced = reduce_monomial(mono)
        for r_mono, r_coeff in reduced:
            result[r_mono] += coeff * r_coeff
    return Polynomial(dict(result))


#INTERSECTION COMPUTATION

def compute_intersection(
    basis1: List[Polynomial],
    g1: Polynomial,
    basis2: List[Polynomial],
    g2: Polynomial,
    rank_method: str = "numpy",
    return_matrix: bool = False,
    tolerance: float = 1e-8,
    verbose: bool = True
) -> Union[Tuple[bool, List[Polynomial], Dict[str, any]], np.ndarray]:
    """
    Compute the intersection of span{basis1 * g1} and span{basis2 * g2}.
    
    This solves a homogeneous linear system by:
    1. Expanding all products basis[i] * g
    2. Building coefficient matrix (rows = monomials, cols = products)
    3. Computing rank using specified method (NumPy or LinBox)
    4. Determining nullity using rank-nullity theorem
    5. Extracting nullspace vectors via SVD and reconstructing intersection basis
    
    Key insight: For a matrix M with shape (m, n):
        nullity = n - rank
    where n is the number of columns (number of basis products).
    
    Args:
        basis1: First basis (list of Polynomials)
        g1: First element to multiply with
        basis2: Second basis (list of Polynomials)
        g2: Second element to multiply with
        rank_method: "numpy" or "linbox" for rank computation
        return_matrix: If True, return coefficient matrix instead of intersection
        tolerance: Threshold for singular values (numpy only)
        verbose: If True, print progress information
        
    Returns:
        Tuple (intersection_exists, intersection_basis, debug_info) or coefficient matrix
        where intersection_basis is a list of Polynomial objects forming a basis for the intersection
    """
    if verbose:
        print("=" * 70)
        print("Computing Intersection")
        print("=" * 70)
        print(f"  Basis 1: {len(basis1)} elements")
        print(f"  Basis 2: {len(basis2)} elements")
        print(f"  Rank method: {rank_method}")
    
    # Step 1: Compute all expansions
    expansions = []
    
    if verbose:
        print("  Expanding basis1 * g1...")
    for poly in tqdm(basis1, disable=not verbose, desc="Basis1"):
        expansions.append(fully_expand(poly * g1))
    
    if verbose:
        print("  Expanding basis2 * g2...")
    for poly in tqdm(basis2, disable=not verbose, desc="Basis2"):
        # Note: we add negative so the system is basis1*g1 - basis2*g2 = 0
        expansions.append(-fully_expand(poly * g2))
    
    # Step 2: Collect all monomials (these are the ROWS of the matrix)
    if verbose:
        print("  Collecting monomials...")
    
    all_monomials: set[Monomial] = set()
    for expansion in expansions:
        all_monomials.update(expansion.terms.keys())
    
    monomial_list = sorted(all_monomials)
    
    num_monomials = len(monomial_list)
    num_products = len(expansions)

    if verbose:
        print(f"  Found {num_monomials} unique monomials (matrix rows)")
        print(f"  Number of basis products (matrix cols): {num_products}")
        if num_monomials < 50:
            print(f"  Monomials: {monomial_list}")
    
    if not monomial_list or not expansions:
        if return_matrix:
            return np.array([])
        return False, [], {"intersection_exists": False, "rank": 0, "nullity": 0}
    
    # Step 3: Build sparse coefficient matrix
    if verbose:
        print("  Building coefficient matrix...")
    
    mono_to_idx = {mono: i for i, mono in enumerate(monomial_list)}
    v = []  # values
    r = []  # row indices
    c = []  # column indices
    
    for col, expansion in enumerate(expansions):
        for mono, coeff in expansion.terms.items():
            if coeff != 0:
                v.append(float(coeff))
                r.append(mono_to_idx[mono])
                c.append(col)
    
    # Create sparse matrix
    coo = coo_matrix(
        (np.array(v), (np.array(r), np.array(c))),
        shape=(num_monomials, num_products)
    )
    csr = coo.tocsr()
    
    if verbose:
        nnz = coo.nnz
        total = coo.shape[0] * coo.shape[1]
        sparsity = (1.0 - (nnz / total)) * 100
        print(f"  Sparse matrix: {coo.shape[0]} rows × {coo.shape[1]} cols")
        print(f"  Non-zeros: {nnz} / {total} ({sparsity:.2f}% sparse)")
    
    if return_matrix:
        return coo.toarray()
    
    # Step 4: Compute rank using specified method
    if verbose:
        print("\n[RANK COMPUTATION]")
    
    try:  
        computer = RankComputer(backend=rank_method)
        rank, rank_info = computer.compute_rank(
        csr,
        tol=tolerance,
        verbose=verbose
        )
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Rank computation failed: {e}")
        return False, [], {"intersection_exists": False, "rank": 0, "nullity": num_products, "error": str(e)}
# Step 5a: Method 1 - Rank-Nullity Theorem
    nullity_from_rank = num_products - rank

    if verbose:
        print(f"\n[NULLITY COMPUTATION]")
        print(f"  Method 1 (Rank-Nullity Theorem):")
        print(f"    num_products (columns) = {num_products}")
        print(f"    rank(M) = {rank}")
        print(f"    nullity = num_products - rank = {num_products} - {rank} = {nullity_from_rank}")

# Step 5b: Method 2 - Direct Nullspace
    vect = scipy.linalg.null_space(coo.toarray(), rcond=tolerance)
    nullity_from_nullspace = vect.shape[1]

    if verbose:
        print(f"\n  Method 2 (Direct Nullspace):")
        print(f"    scipy.linalg.null_space() returned matrix of shape: {vect.shape}")
        print(f"    nullity = vect.shape[1] = {nullity_from_nullspace}")

# Step 5c: Compare and diagnose
    if verbose:
        print(f"\n[COMPARISON]")
        print(f"  Nullity from rank-nullity theorem: {nullity_from_rank}")
        print(f"  Nullity from null space dimension: {nullity_from_nullspace}")
    
    if nullity_from_rank == nullity_from_nullspace:
        print(f"  ✓ METHODS AGREE")
    else:
        print(f"  ✗ DISCREPANCY DETECTED")
        print(f"    Difference: {abs(nullity_from_rank - nullity_from_nullspace)}")
        print(f"    This suggests tolerance issues or numerical instability.")

# Use the direct nullspace dimension (more reliable)
    nullity = nullity_from_nullspace
    intersection_exists = nullity > 0

    if verbose:
        print(f"\n[RESULT]")
    if intersection_exists:
        print(f"  ✓ INTERSECTION EXISTS")
        print(f"    Dimension of intersection: {nullity}")
    else:
        print(f"  ✗ NO NON-TRIVIAL INTERSECTION")
    
    # Step 6: Compute SVD to extract nullspace
    if verbose:
        print(f"\n[SVD NULLSPACE EXTRACTION]")
        print(f"  Computing SVD for nullspace...")
    
    if not intersection_exists:
        # No nullspace to compute
        if verbose:
            print("  No nullspace (rank equals number of columns)")
    else:
        try:
            vect=scipy.linalg.null_space(coo.toarray())
            print(f"Dimension of the null space is {vect.shape[1]}.")

            # Verify nullspace
            prod = coo.toarray() @ vect
            max_residual = np.max(np.abs(prod))
            if verbose:
                print(f"  Verification: max |M @ nullspace| = {max_residual:.2e}")
            
            if not np.allclose(prod, np.zeros(prod.shape), atol=tolerance):
                if verbose:
                    print(f"  [WARNING] SVD nullspace verification failed (residual: {max_residual})")
                    print(f"  Proceeding anyway, but results may be inaccurate")
            
            # Step 7: Reconstruct intersection elements from nullspace
            if verbose:
                print(f"\n[RECONSTRUCTING INTERSECTION BASIS]")
            intersection_basis = []
            n1=len(basis1)

            for col_idx in range(vect.shape[1]):
                null_vector = vect[:, col_idx]

                int_coeffs = rationalize_vector(null_vector)
                
                if verbose:
                    print(f"    Rationalized coefficients: {int_coeffs}")
                
                # Build linear combination from basis1 side
                result = Polynomial.zero()
                for i in range(n1):
                    if int_coeffs[i] != 0:
                        result = result + (int_coeffs[i] * fully_expand(basis1[i] * g1))
                    if not result.is_zero():
                        intersection_basis.append(result)
                    if verbose:
                        return result
                else:
                    if verbose:
                        print(f"    Resulted in zero polynomial, skipping")
            
            if verbose:
                print(f"  Final intersection dimension: {len(intersection_basis)}")

            """
            # For nullspace extraction, we need k < min(m, n) to get null vectors
            # svds computes the k largest singular values by default
            # We want the SMALLEST singular values, so use which="SM"
            k = min(num_monomials, num_products) - 1
            if k < 1:
                k = 1
            
            U, s, Vt = svds(csr, k=k, which="SM")
            
            # Extract nullspace vectors (those with near-zero singular values)
            # s is 1D array of singular values (already sorted, smallest first)
            # Vt has shape (k, num_products) - Vt[i] is right singular vector for s[i]
            # We need the vectors where s < tolerance
            
            null_mask = s < tolerance
            
            if verbose:
                print(f"  SVD computed: {len(s)} singular values")
                print(f"  Singular values (smallest): {s}")
                print(f"  Null space dimension (s < {tolerance}): {np.sum(null_mask)}")
            
            # Vt[null_mask, :] gives rows corresponding to near-zero singular values
            # These are already the right singular vectors we need
            null_space = Vt[null_mask, :].T  # Transpose to get (num_products, nullity)
            
            if verbose:
                print(f"  Nullspace shape: {null_space.shape}")
            
            # Verify nullspace
            prod = csr @ null_space
            max_residual = np.max(np.abs(prod))
            if verbose:
                print(f"  Verification: max |M @ nullspace| = {max_residual:.2e}")
            
            if not np.allclose(prod, np.zeros(prod.shape), atol=tolerance):
                if verbose:
                    print(f"  [WARNING] SVD nullspace verification failed (residual: {max_residual})")
                    print(f"  Proceeding anyway, but results may be inaccurate")
            
            # Step 7: Reconstruct intersection elements from nullspace
            if verbose:
                print(f"\n[RECONSTRUCTING INTERSECTION BASIS]")
            
            intersection_basis = []
            n1 = len(basis1)
            
            for col_idx in range(null_space.shape[1]):
                if verbose:
                    print(f"  Processing nullspace vector {col_idx + 1}...")
                
                null_vector = null_space[:, col_idx]
                int_coeffs = rationalize_vector(null_vector)
                
                if verbose:
                    print(f"    Rationalized coefficients: {int_coeffs}")
                
                # Build linear combination from basis1 side
                result = Polynomial.zero()
                for i in range(n1):
                    if int_coeffs[i] != 0:
                        result = result + (int_coeffs[i] * fully_expand(basis1[i] * g1))
                
                if not result.is_zero():
                    intersection_basis.append(result)
                    if verbose:
                        print(f"    Added to basis: {result}")
                else:
                    if verbose:
                        print(f"    Resulted in zero polynomial, skipping")
            
            if verbose:
                print(f"  Final intersection dimension: {len(intersection_basis)}")
        """
        except Exception as e:
            if verbose:
                print(f"  [ERROR] SVD nullspace extraction failed: {e}")
            intersection_basis = []
    
            
    if verbose:
        print("=" * 70)
    
    debug_info = {
        "intersection_exists": intersection_exists,
        "rank": rank,
        "nullity": nullity,
        "num_monomials": num_monomials,
        "num_products": num_products,
        "rank_computation": rank_info,
        "intersection_dimension": len(intersection_basis)
    }
    
    return intersection_exists, intersection_basis, debug_info

#MAIN DEMONSTRATION

def main():
    """Demonstration of the non-commutative algebra system."""
    print("=" * 70)
    print("Non-Commutative Algebra System with Rank Computation")
    print("Rule: e_n * e_m = e_m * e_n + (m-n)*e_{n+m} when n > m")
    print("=" * 70)
    print()
    
    # Basic examples
    print("Basic Examples:")
    print(f"e(2) * e(1) = {e(2) * e(1)}")
    print(f"e(3) * e(1) = {e(3) * e(1)}")
    print(f"e(3) * e(2) * e(1) = {e(3) * e(2) * e(1)}")
    print()
    
    # g function
    print("Special g functions:")
    g22 = g(2, 2)
    g23 = g(2, 3)

    print(f"g(2,2) = {g22}")
    print(f"g(2,3) = {g23}")
    print()
    
    # Basis generation
    print("=" * 70)
    print("Basis Generation:")
    print("=" * 70)
    print()
    
    print("UW_basis(5):")
    uw5 = generate_uw_basis(5)
    for i, elem in enumerate(uw5):
        print(f"  [{i}] {elem}")
    print()
    
    # Intersection computation
    print("=" * 70)
    print("Intersection Computation:")
    print("=" * 70)
    print()
    
    # Test with small degrees
    for target_degree in range(31, 33):
        print(f"\nTesting Degree {target_degree + 5}:")
        print("-" * 70)
        
        # Compute intersection
        intersection_exists, intersection_basis, debug_info = compute_intersection(
            generate_uw_basis(target_degree+1),
            g22,
            generate_uw_basis(target_degree),
            g23,
            rank_method="numpy",
            verbose=True
        )
        print(f"Degree {target_degree + 5}: exists={intersection_exists}")


if __name__ == "__main__":
    main()