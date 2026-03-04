"""
Optimized non-commutative algebra implementation for computing intersections
of U(W_+) subspaces with non-multiplication rules.

- Polynomial representation with operator overloading
- Rank computation using NumPy, SymPy, and LinBox backends
- Dual null space computation with safety verification

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
import scipy as sp
from scipy.sparse import coo_matrix, issparse, csr_matrix
from scipy.sparse.linalg import svds
from collections import defaultdict
from tqdm import tqdm
import warnings

Monomial = Tuple[int, ...]  # Ordered tuple of indices representing e_{i1} * e_{i2} * ...
CoeffDict = Dict[Monomial, int]  # Sparse representation: monomial -> coefficient


#RANK COMPUTATION SYSTEM

class RankComputer:
    """
    Compute matrix rank and null space dimension using different backends.
    Supports NumPy, SymPy, and LinBox for rank computation.
    """
    
    def __init__(self, backend: str = "numpy"):
        """
        Initialize rank computer.
        
        Args:
            backend: Either "numpy", "sympy", or "linbox"
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
        elif self.backend == "sympy":
            try:
                from sympy import Matrix
                self.Matrix = Matrix
            except ImportError:
                raise ImportError(
                    "SymPy not installed. Install with: pip install sympy "
                    "or use backend='numpy' instead."
                )
        elif self.backend != "numpy":
            raise ValueError(f"Unknown backend: {backend}. Choose 'numpy', 'sympy', or 'linbox'.")
    
    def rank_sympy(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        verbose: bool = False
    ) -> Tuple[int, Dict[str, any]]:
        """
        Compute rank and null space dimension using SymPy.
        
        SymPy's nullspace() method returns basis vectors for the null space,
        allowing direct calculation of nullity = number of null space vectors.
        
        Args:
            matrix: Input matrix (dense array or sparse CSR)
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
            print(f"  Using SymPy for null space computation")
        
        # Convert to SymPy Matrix
        sympy_matrix = self.Matrix(matrix_dense.tolist())
        
        # Get rank using SymPy
        rank = sympy_matrix.rank()
        
        # Get null space basis vectors
        nullspace_basis = sympy_matrix.nullspace()
        
        # Null space dimension = number of basis vectors
        nullity = len(nullspace_basis)
        
        # Verify rank-nullity theorem
        num_cols = sympy_matrix.cols
        expected_nullity = num_cols - rank
        
        if verbose:
            print(f"  Rank (SymPy): {rank}")
            print(f"  Null space dimension: {nullity}")
            print(f"  Number of columns: {num_cols}")
            print(f"  Rank-nullity check: {rank} + {nullity} = {rank + nullity}")
            print(f"  Rank-nullity theorem verified: {rank + nullity == num_cols}")
        
        debug_info = {
            "method": "sympy_nullspace",
            "rank": rank,
            "shape": matrix_dense.shape,
            "nullity": nullity,
            "nullspace_basis_vectors": len(nullspace_basis),
            "rank_nullity_verified": rank + nullity == num_cols,
            "expected_nullity": expected_nullity
        }
        
        return rank, debug_info
    
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
                    if val != 0:
                        A[i, j] = int(val) if val == int(val) else val
            
            # Compute rank
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
    
    def compute_nullity_from_rank_nullity_theorem(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        rank: int,
        verbose: bool = False
    ) -> Tuple[int, Dict[str, any]]:
        """
        Compute null space dimension using the rank-nullity theorem.
        
        Theorem: For a matrix with m rows and n columns:
            rank(A) + nullity(A) = n
            Therefore: nullity = n - rank
        
        Args:
            matrix: Input matrix
            rank: The rank of the matrix
            verbose: Print debug information
            
        Returns:
            Tuple of (nullity, debug_info_dict)
        """
        if isinstance(matrix, csr_matrix):
            num_cols = matrix.shape[1]
        else:
            num_cols = np.asarray(matrix).shape[1]
        
        nullity = num_cols - rank
        
        if verbose:
            print(f"  Rank-Nullity Theorem: nullity = columns - rank")
            print(f"  nullity = {num_cols} - {rank} = {nullity}")
        
        debug_info = {
            "method": "rank_nullity_theorem",
            "rank": rank,
            "num_columns": num_cols,
            "nullity": nullity,
            "formula": "nullity = num_columns - rank"
        }
        
        return nullity, debug_info
    
    def compute_rank_and_compare_nullity(
        self,
        matrix: Union[np.ndarray, csr_matrix],
        backend: str = "numpy",
        verbose: bool = True
    ) -> Tuple[int, int, Dict[str, any]]:
        """
        Compute nullity using TWO methods and compare for safety:
        
        Method 1: Direct SymPy null space computation
        Method 2: Rank-Nullity theorem (nullity = n - rank)
        
        This dual approach ensures correctness by verifying both methods agree.
        
        Args:
            matrix: Input matrix
            backend: Backend to use for rank computation ("numpy" or "sympy")
            verbose: Print detailed comparison results
            
        Returns:
            Tuple of (rank, nullity, comparison_info_dict)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("DUAL NULL SPACE COMPUTATION WITH SAFETY CHECK")
            print("=" * 70)
        
        # Convert sparse to dense if needed for shape info
        if isinstance(matrix, csr_matrix):
            num_cols = matrix.shape[1]
        else:
            num_cols = np.asarray(matrix).shape[1]
        
        if verbose:
            print(f"Matrix dimensions: {matrix.shape}")
            print(f"Number of columns: {num_cols}\n")
        
        # METHOD 1: Direct SymPy null space computation
        if verbose:
            print("[METHOD 1] Direct SymPy Null Space Computation")
            print("-" * 70)
        
        try:
            computer_sympy = RankComputer(backend="sympy")
            rank_sympy, info_sympy = computer_sympy.compute_rank(matrix, verbose=verbose)
            nullity_direct = info_sympy["nullity"]
            method1_success = True
            
            if verbose:
                print(f"Method 1: nullity = {nullity_direct}\n")
        except Exception as e:
            if verbose:
                print(f"Method 1 failed: {e}\n")
            nullity_direct = None
            method1_success = False
        
        # METHOD 2: Rank-Nullity theorem
        if verbose:
            print("[METHOD 2] Rank-Nullity Theorem")
            print("-" * 70)
        
        try:
            # Compute rank using specified backend
            computer_rank = RankComputer(backend=backend)
            rank_computed, info_rank = computer_rank.compute_rank(matrix, verbose=verbose)
            
            # Apply rank-nullity theorem
            nullity_theorem, info_theorem = self.compute_nullity_from_rank_nullity_theorem(
                matrix, rank_computed, verbose=verbose
            )
            method2_success = True
            
            if verbose:
                print(f"Method 2: nullity = {nullity_theorem}\n")
        except Exception as e:
            if verbose:
                print(f"Method 2 failed: {e}\n")
            nullity_theorem = None
            method2_success = False
        
        # COMPARISON AND VERIFICATION
        if verbose:
            print("=" * 70)
            print("COMPARISON AND VERIFICATION")
            print("=" * 70)
        
        comparison_info = {
            "method1_success": method1_success,
            "method2_success": method2_success,
            "nullity_direct_sympy": nullity_direct,
            "nullity_from_theorem": nullity_theorem,
            "rank": rank_computed if method2_success else (rank_sympy if method1_success else None),
            "num_columns": num_cols
        }
        
        if method1_success and method2_success:
            match = nullity_direct == nullity_theorem
            comparison_info["methods_agree"] = match
            
            if verbose:
                print(f"Method 1 (SymPy nullspace):    nullity = {nullity_direct}")
                print(f"Method 2 (Rank-Nullity thm):  nullity = {nullity_theorem}")
                print()
                
                if match:
                    print(f"RESULTS MATCH - Both methods agree!")
                    print(f"  VERIFIED NULL SPACE DIMENSION: {nullity_direct}")
            
                else:
                    print(f"ESULTS DIFFER - Safety check FAILED!")
                    print(f"  Difference: {abs(nullity_direct - nullity_theorem)}")
        
    
        if verbose:
            print("=" * 70 + "\n")
        
        # Return the best available result
        final_rank = rank_computed if method2_success else (rank_sympy if method1_success else None)
        final_nullity = nullity_direct if nullity_direct is not None else nullity_theorem
        
        return final_rank, final_nullity, comparison_info
    
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
        elif self.backend == "sympy":
            return self.rank_sympy(matrix, **kwargs)
        elif self.backend == "linbox":
            return self.rank_linbox(matrix, **kwargs)


def compute_rank_with_comparison(
    matrix: Union[np.ndarray, csr_matrix],
    verbose: bool = True,
    try_linbox: bool = False
) -> Tuple[int, Dict[str, any]]:
    """
    Compute rank and optionally compare multiple methods.
    
    Args:
        matrix: Input matrix
        verbose: Print comparison results
        try_linbox: Also try LinBox if available
        
    Returns:
        Tuple of (rank, comprehensive_debug_info)
    """
    results = {}
    
    # Always compute with NumPy
    numpy_computer = RankComputer(backend="numpy")
    rank_np, info_np = numpy_computer.rank_numpy(matrix, verbose=verbose)
    results["numpy"] = (rank_np, info_np)
    
    # Optionally compute with LinBox
    if try_linbox:
        try:
            linbox_computer = RankComputer(backend="linbox")
            if isinstance(matrix, csr_matrix) and matrix.shape[0] * matrix.shape[1] > 1e7:
                if verbose:
                    print("  [WARNING] Matrix too large for LinBox, skipping...")
            else:
                rank_lb, info_lb = linbox_computer.rank_linbox(matrix, verbose=verbose)
                results["linbox"] = (rank_lb, info_lb)
        except ImportError:
            if verbose:
                print("  [NOTE] LinBox not available, skipping LinBox computation")
        except Exception as e:
            if verbose:
                print(f"  [WARNING] LinBox computation failed: {e}")
    
    # Compile comparison
    all_ranks = [rank for rank, _ in results.values()]
    
    if verbose:
        print("\n[RANK COMPUTATION SUMMARY]")
        for backend, (rank, info) in results.items():
            print(f"  {backend:10s}: rank = {rank}")
        
        if len(set(all_ranks)) > 1:
            print(f"  [WARNING] Results differ! Verify computation.")
    
    combined_info = {
        "primary_rank": rank_np,
        "methods": results,
        "consensus": len(set(all_ranks)) == 1
    }
    
    return rank_np, combined_info


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
    verbose: bool = True,
    use_safety_check: bool = True
) -> Union[Tuple[bool, List[Polynomial], Dict[str, any]], np.ndarray]:
    """
    Compute the intersection of span{basis1 * g1} and span{basis2 * g2}.
    
    This solves a homogeneous linear system by:
    1. Expanding all products basis[i] * g
    2. Building coefficient matrix (rows = monomials, cols = products)
    3. Computing rank using specified method (NumPy, SymPy, or LinBox)
    4. Determining nullity using rank-nullity theorem with optional safety verification
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
        use_safety_check: If True, use dual-method null space verification
        
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
        print(f"  Safety check: {'ENABLED' if use_safety_check else 'DISABLED'}")
    
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
    
    # Step 4: Compute rank and nullity with optional safety check
    if verbose:
        print("\n[RANK AND NULL SPACE COMPUTATION]")
    
    if use_safety_check:
        try:
            computer = RankComputer(backend="numpy")
            rank, nullity, comparison_info = computer.compute_rank_and_compare_nullity(
                csr,
                backend=rank_method,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Safety check failed: {e}")
                print(f"  Falling back to single method...")
            
            try:
                computer = RankComputer(backend=rank_method)
                rank, rank_info = computer.compute_rank(
                    csr,
                    tol=tolerance if rank_method == "numpy" else None,
                    verbose=verbose
                )
                nullity = num_products - rank
                comparison_info = {"safety_check_failed": True, "rank_info": rank_info}
            except Exception as e2:
                if verbose:
                    print(f"  [ERROR] Rank computation failed: {e2}")
                return False, [], {"intersection_exists": False, "rank": 0, "nullity": num_products, "error": str(e2)}
    else:
        try:
            computer = RankComputer(backend=rank_method)
            rank, rank_info = computer.compute_rank(
                csr,
                tol=tolerance if rank_method == "numpy" else None,
                verbose=verbose
            )
            nullity = num_products - rank
            comparison_info = rank_info
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Rank computation failed: {e}")
            return False, [], {"intersection_exists": False, "rank": 0, "nullity": num_products, "error": str(e)}
    
    intersection_exists = nullity > 0
    
    if verbose:
        print(f"\n[RESULT]")
        if intersection_exists:
            print(f"  ✓ INTERSECTION EXISTS")
            print(f"    Dimension of intersection: {nullity}")
        else:
            print(f"  ✗ NO NON-TRIVIAL INTERSECTION")
            print(f"    (Only trivial solution: all coefficients = 0)")
    
    # Step 5: Compute SVD to extract nullspace
    if verbose:
        print(f"\n[SVD NULLSPACE EXTRACTION]")
        print(f"  Computing SVD for nullspace (k={min(coo.shape)-1})...")
    
    if not intersection_exists:
        # No nullspace to compute
        intersection_basis = []
        if verbose:
            print("  No nullspace (rank equals number of columns)")
    else:
        try:
            # Compute SVD: keep all but the rank singular vectors
            k = min(coo.shape) - 1
            U, s, Vt = svds(coo, k=k, which="SM")
            
            # Extract nullspace vectors (those with near-zero singular values)
            null_mask = s < tolerance
            null_space = Vt.T[:, null_mask]
            
            if verbose:
                print(f"  SVD computed: {len(s)} singular values")
                print(f"  Singular values (smallest): {s[:5]}")
                print(f"  Nullspace dimension: {null_space.shape[1]}")
            
            # Verify nullspace
            prod = coo @ null_space
            max_residual = np.max(np.abs(prod))
            if verbose:
                print(f"  Verification: max |M @ nullspace| = {max_residual:.2e}")
            
            if not np.allclose(prod, np.zeros(prod.shape), atol=tolerance):
                if verbose:
                    print(f"  [WARNING] SVD nullspace verification failed (residual: {max_residual})")
                    print(f"  Proceeding anyway, but results may be inaccurate")
            
            # Step 6: Reconstruct intersection elements from nullspace
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
        "rank_computation": comparison_info,
        "intersection_dimension": len(intersection_basis),
        "safety_check_enabled": use_safety_check
    }
    
    return intersection_exists, intersection_basis, debug_info

#MAIN DEMONSTRATION

def main():
    """Demonstration of the non-commutative algebra system."""
    print("=" * 70)
    print("Non-Commutative Algebra System with Dual Null Space Verification")
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
    print("Intersection Computation with Safety Verification:")
    print("=" * 70)
    print()
    
    # Test with small degrees
    for target_degree in range(20, 25):
        print(f"\nTesting Degree {target_degree + 4}:")
        print("-" * 70)
        intersection_exists, basis, debug_info = compute_intersection(
            generate_uw_basis(target_degree + 1),
            e(4),
            generate_uw_basis(target_degree),
            e(5),
            rank_method="numpy",
            verbose=True,
            use_safety_check=True
        )
        print(f"Degree {target_degree + 4}: exists={intersection_exists}")


if __name__ == "__main__":
    main()