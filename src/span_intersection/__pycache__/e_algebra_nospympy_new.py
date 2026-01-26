# Optimized Non-Commutative Algebra with COO Sparse Matrix Format
# Uses scipy.sparse.coo_matrix for true sparse computation

from __future__ import annotations
from typing import Tuple, Dict, Iterator, List, Optional, Union
from functools import reduce
from dataclasses import dataclass, field
from math import gcd
from fractions import Fraction
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from collections import defaultdict
import gc

Monomial = Tuple[int, ...]
CoeffDict = Dict[Monomial, int]


# ============================================================================
# OPTIMIZATION 1: Minimal Cache with Smart Cleanup
# ============================================================================

_reduce_cache: Dict[Monomial, Tuple[Tuple[Monomial, int], ...]] = {}
_cache_access_count = defaultdict(int)

def _cleanup_cache(target_size: int = 5000):
    """Keep only most-used cache entries."""
    global _reduce_cache
    if len(_reduce_cache) > target_size:
        keep_count = int(target_size * 0.8)
        sorted_by_access = sorted(
            _reduce_cache.items(),
            key=lambda x: _cache_access_count[x[0]],
            reverse=True
        )
        _reduce_cache = dict(sorted_by_access[:keep_count])
        gc.collect()


def reduce_monomial(seq: Monomial) -> Tuple[Tuple[Monomial, int], ...]:
    """
    Reduce monomial using non-commutative rule with minimal caching.
    """
    if seq in _reduce_cache:
        _cache_access_count[seq] += 1
        return _reduce_cache[seq]
    
    if len(seq) <= 1:
        result = ((seq, 1),) if seq else (((), 1),)
        _reduce_cache[seq] = result
        return result
    
    if all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)):
        result = ((seq, 1),)
        _reduce_cache[seq] = result
        return result
    
    for i in range(len(seq) - 1):
        n, m = seq[i], seq[i + 1]
        if n > m:
            left = seq[:i]
            right = seq[i + 2:]
            
            swapped = left + (m, n) + right
            term1_pairs = reduce_monomial(swapped)
            
            correction_coeff = m - n
            corrected = left + (n + m,) + right
            term2_pairs = reduce_monomial(corrected)
            
            result_dict: Dict[Monomial, int] = {}
            for mono, coeff in term1_pairs:
                result_dict[mono] = result_dict.get(mono, 0) + coeff
            for mono, coeff in term2_pairs:
                result_dict[mono] = result_dict.get(mono, 0) + correction_coeff * coeff
            
            result = tuple((m, c) for m, c in result_dict.items() if c != 0)
            _reduce_cache[seq] = result
            
            if len(_reduce_cache) % 2000 == 0:
                _cleanup_cache(5000)
            
            return result
    
    return ((seq, 1),)


def reduce_to_dict(seq: Monomial) -> CoeffDict:
    """Convert cached result to dictionary."""
    return {mono: coeff for mono, coeff in reduce_monomial(seq)}


# ============================================================================
# POLYNOMIAL CLASS
# ============================================================================

@dataclass
class Polynomial:
    """Sparse polynomial representation."""
    terms: CoeffDict = field(default_factory=dict)
    
    def __post_init__(self):
        self.terms = {m: c for m, c in self.terms.items() if c != 0}
    
    @classmethod
    def zero(cls) -> Polynomial:
        return cls({})
    
    @classmethod
    def constant(cls, value: int) -> Polynomial:
        return cls({(): value}) if value != 0 else cls.zero()
    
    @classmethod
    def from_monomial(cls, monomial: Monomial, coeff: int = 1) -> Polynomial:
        if coeff == 0:
            return cls.zero()
        return cls({tuple(monomial): coeff})
    
    @classmethod
    def basis_element(cls, index: int) -> Polynomial:
        return cls.from_monomial((index,), 1)
    
    def __add__(self, other: Polynomial) -> Polynomial:
        result = dict(self.terms)
        for mono, coeff in other.terms.items():
            result[mono] = result.get(mono, 0) + coeff
        return Polynomial(result)
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
    
    def __neg__(self) -> Polynomial:
        return Polynomial({m: -c for m, c in self.terms.items()})
    
    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + (-other)
    
    def __mul__(self, other: Union[Polynomial, int]) -> Polynomial:
        if isinstance(other, int):
            if other == 0:
                return Polynomial.zero()
            return Polynomial({m: c * other for m, c in self.terms.items()})
        
        if isinstance(other, Polynomial):
            if not self.terms or not other.terms:
                return Polynomial.zero()
            
            result: Dict[Monomial, int] = defaultdict(int)
            for m1, c1 in self.terms.items():
                for m2, c2 in other.terms.items():
                    concat = m1 + m2
                    reduced = reduce_to_dict(concat)
                    for mono, mono_coeff in reduced.items():
                        result[mono] += c1 * c2 * mono_coeff
            
            return Polynomial(dict(result))
        
        return NotImplemented
    
    def __rmul__(self, other):
        if isinstance(other, int):
            return self.__mul__(other)
        return NotImplemented
    
    def __pow__(self, exponent: int) -> Polynomial:
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Only non-negative integer powers supported")
        
        if exponent == 0:
            return Polynomial.constant(1)
        if exponent == 1:
            return Polynomial(dict(self.terms))
        
        result = Polynomial.constant(1)
        base = Polynomial(dict(self.terms))
        
        while exponent > 0:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent >>= 1
        
        return result
    
    def is_zero(self) -> bool:
        return len(self.terms) == 0
    
    def degree(self) -> int:
        if not self.terms:
            return -1
        return max(sum(mono) for mono in self.terms.keys())
    
    def num_terms(self) -> int:
        return len(self.terms)
    
    def __repr__(self) -> str:
        if not self.terms:
            return "Polynomial({})"
        return f"Polynomial({dict(self.terms)})"
    
    def __str__(self) -> str:
        if not self.terms:
            return "0"
        
        sorted_items = sorted(self.terms.items(), key=lambda kv: (len(kv[0]), kv[0]))
        
        parts = []
        for mono, coeff in sorted_items:
            if mono == ():
                parts.append(str(coeff))
            else:
                mono_str = "*".join(f"e_{i}" for i in mono)
                if coeff == 1:
                    parts.append(mono_str)
                elif coeff == -1:
                    parts.append(f"-{mono_str}")
                else:
                    parts.append(f"{coeff}*{mono_str}")
        
        result = parts[0]
        for part in parts[1:]:
            if part.startswith("-"):
                result += f" - {part[1:]}"
            else:
                result += f" + {part}"
        
        return result
    
    def __bool__(self):
        return not self.is_zero()
    
    def copy(self) -> Polynomial:
        return Polynomial(dict(self.terms))
    
    def expand(self) -> Polynomial:
        return self.copy()


def e(index: int) -> Polynomial:
    """Create basis element e_i."""
    return Polynomial.basis_element(index)


def g(n: int, m: int) -> Polynomial:
    """Construct g(n,m) = e_n * e_m - e_1 * e_{n+m-1} + (n-1) * e_{n+m}"""
    term1 = e(n) * e(m)
    term2 = e(1) * e(n + m - 1)
    term3 = (n - 1) * e(n + m)
    return term1 - term2 + term3


# ============================================================================
# BASIS GENERATION
# ============================================================================

def integer_partitions(n: int, min_part: int = 1) -> Iterator[Tuple[int, ...]]:
    """Generate all integer partitions of n."""
    yield (n,)
    for i in range(min_part, n // 2 + 1):
        for partition in integer_partitions(n - i, i):
            yield (i,) + partition


def generate_uw_basis(n: int, verbose: bool = False) -> List[Polynomial]:
    """Generate PBW basis for U(W_+)_n."""
    if verbose:
        print(f"Generating UW_basis({n})...")
    
    partitions = list(integer_partitions(n))
    basis = []
    
    for partition in reversed(partitions):
        product = Polynomial.constant(1)
        for index in partition:
            product = product * e(index)
        basis.append(product)
    
    if verbose:
        print(f"[GEN] Finished generate_uw_basis({n}): {len(basis)} elements")
    
    return basis


# ============================================================================
# OPTIMIZATION 2: COO Format Null Space Solver
# ============================================================================

def compute_null_space_coo(A_coo: coo_matrix, 
                           tolerance: float = 1e-10,
                           verbose: bool = True) -> np.ndarray:
    """
    Compute null space of sparse COO matrix using eigenvalue decomposition.
    
    COO (Coordinate) format is perfect for construction and direct computation.
    
    Method: null space of A = eigenvectors of A^T*A with eigenvalue ~0
    This works for any matrix shape and is memory-efficient.
    
    Args:
        A_coo: Sparse matrix in COO format (m x n)
        tolerance: Threshold for considering singular value as zero
        verbose: Print progress
    
    Returns:
        Null space basis as array with shape (n, null_dimension)
    """
    
    m, n = A_coo.shape
    nnz = A_coo.nnz
    
    if verbose:
        print(f"[NULL] Computing null space of ({m} x {n}) COO matrix")
        print(f"[NULL] Non-zeros: {nnz}, Sparsity: {100 * (1 - nnz/(m*n)):.1f}%")
    
    if nnz == 0:
        if verbose:
            print(f"[NULL] Matrix is empty, null space is full")
        return np.eye(n)
    
    # Convert COO to CSR for efficient matrix operations (A^T * A)
    # COO is good for construction, CSR is better for multiplication
    if verbose:
        print(f"[NULL] Converting COO to CSR for A^T*A computation...")
    
    A_csr = A_coo.tocsr()
    
    # Compute Gram matrix A^T * A
    # This is the key insight: null(A) = eigenvectors of A^T*A with eigenvalue 0
    if verbose:
        print(f"[NULL] Computing A^T * A (Gram matrix)...")
    
    ATA = A_csr.T @ A_csr
    
    if verbose:
        print(f"[NULL] Gram matrix computed: {ATA.nnz} non-zeros")
        print(f"[NULL] Computing smallest eigenvalues...")
    
    # Request smallest eigenvalues using sparse eigenvalue solver
    n_wanted = min(20, n - 1)  # Request up to 20 eigenvectors
    
    try:
        eigenvalues, eigenvectors = eigsh(ATA, 
                                          k=n_wanted, 
                                          which='SM',  # Smallest magnitude
                                          return_eigenvectors=True,
                                          maxiter=n * 10)
        
        # Find null space vectors (eigenvectors with eigenvalue ~0)
        null_mask = eigenvalues < tolerance
        null_dim = np.sum(null_mask)
        
        if verbose:
            print(f"[NULL] Eigenvalues (smallest {min(5, len(eigenvalues))}): {eigenvalues[:min(5, len(eigenvalues))]}")
            print(f"[NULL] Null dimension: {null_dim}")
        
        if null_dim > 0:
            null_space = eigenvectors[:, null_mask]
            return null_space
        else:
            return np.array([]).reshape(n, 0)
    
    except Exception as e:
        if verbose:
            print(f"[NULL] Eigenvalue solver failed: {e}")
            print(f"[NULL] Attempting dense SVD fallback...")
        
        # Fallback to dense SVD for smaller matrices
        if m * n < 10_000_000:
            if verbose:
                print(f"[NULL] Converting to dense and computing SVD...")
            
            A_dense = A_coo.toarray()
            U, s, Vt = np.linalg.svd(A_dense, full_matrices=True)
            
            # Find near-zero singular values
            null_mask = s < tolerance
            null_dim = np.sum(null_mask)
            
            if verbose:
                print(f"[NULL] Singular values (smallest {min(5, len(s))}): {s[-min(5, len(s)):]}")
                print(f"[NULL] Null dimension: {null_dim}")
            
            if null_dim > 0:
                return Vt.T[:, -null_dim:]
            else:
                return np.array([]).reshape(n, 0)
        else:
            raise RuntimeError(f"Matrix too large ({m}x{n}) and eigenvalue solver failed")



# ============================================================================
# INTERSECTION COMPUTATION
# ============================================================================

def compute_intersection(
    basis1: List[Polynomial],
    g1: Polynomial,
    basis2: List[Polynomial],
    g2: Polynomial,
    verbose: bool = True
) -> List[Polynomial]:
    """
    Compute intersection using COO sparse matrices.
    Fast and memory-efficient.
    """
    if verbose:
        print("Computing intersection...")
        print(f"  Basis 1: {len(basis1)} elements")
        print(f"  Basis 2: {len(basis2)} elements")
    
    # Step 1: Expand all products and collect monomials
    if verbose:
        print("  Expanding and collecting monomials...")
    
    all_monomials: set[Monomial] = set()
    expansions = []
    
    for i, poly in enumerate(basis1):
        expanded = (poly * g1).expand()
        expansions.append(expanded)
        all_monomials.update(expanded.terms.keys())
        if verbose and (i + 1) % max(1, len(basis1) // 5) == 0:
            print(f"    Expanded {i + 1}/{len(basis1)} from basis1")
    
    for i, poly in enumerate(basis2):
        expanded = (poly * g2).expand()
        expansions.append(expanded)
        all_monomials.update(expanded.terms.keys())
        if verbose and (i + 1) % max(1, len(basis2) // 5) == 0:
            print(f"    Expanded {i + 1}/{len(basis2)} from basis2")
    
    monomial_list = sorted(all_monomials)
    
    if verbose:
        print(f"  Found {len(monomial_list)} unique monomials")
        if len(monomial_list) < 100:
            max_index = max(max(m) if m else 0 for m in monomial_list) if monomial_list else 0
            print(f"  Max index: {max_index}")
    
    if not monomial_list or not expansions:
        return []
    
    # Step 2: Build coefficient matrix in COO format
    if verbose:
        print("  Building coefficient matrix (COO format)...")
    
    mono_to_idx = {mono: i for i, mono in enumerate(monomial_list)}
    
    # COO format: store as lists of (row, col, data)
    row_indices = []
    col_indices = []
    data = []
    
    for col, expansion in enumerate(expansions):
        for mono, coeff in expansion.terms.items():
            if coeff != 0:
                row_indices.append(mono_to_idx[mono])
                col_indices.append(col)
                data.append(float(coeff))
    
    num_rows = len(monomial_list)
    num_cols = len(expansions)
    
    # Create COO matrix
    A_coo = coo_matrix((data, (row_indices, col_indices)), 
                       shape=(num_rows, num_cols), 
                       dtype=np.float64)
    
    print(f"[MAT] COO matrix created: {num_rows} x {num_cols}, {len(data)} non-zeros", flush=True)
    print(f"[MAT] Sparsity: {100*(1 - len(data)/(num_rows*num_cols)):.1f}%", flush=True)
    
    # Step 3: Compute null space using COO matrix
    if verbose:
        print("  Computing null space...")
    
    null_space = compute_null_space_coo(A_coo, verbose=verbose)
    
    if null_space.shape[1] == 0:
        if verbose:
            print("  No null space found!")
        return []
    
    print(f"[NULL] Null space dimension: {null_space.shape[1]}", flush=True)
    
    # Step 4: Reconstruct intersection elements
    if verbose:
        print("  Reconstructing intersection elements...")
    
    intersection_basis = []
    n1 = len(basis1)
    
    for col_idx in range(null_space.shape[1]):
        null_vector = null_space[:, col_idx]
        
        # Rationalize coefficients
        int_coeffs = []
        for val in null_vector:
            if abs(val) < 1e-12:
                int_coeffs.append(0)
            else:
                frac = Fraction(val).limit_denominator(100000)
                int_coeffs.append(int(round(float(frac) * 1e10)) if frac != 0 else 0)
        
        # Reduce by GCD
        non_zero = [abs(x) for x in int_coeffs if x != 0]
        if non_zero:
            common_gcd = reduce(gcd, non_zero)
            int_coeffs = [x // common_gcd for x in int_coeffs]
        
        # Build linear combination from basis1
        result = Polynomial.zero()
        for i in range(n1):
            if int_coeffs[i] != 0:
                result = result + (int_coeffs[i] * (basis1[i] * g1))
        
        if not result.is_zero():
            intersection_basis.append(result.expand())
    
    if verbose:
        print(f"  Intersection dimension: {len(intersection_basis)}")
    
    # Cleanup
    _cleanup_cache(5000)
    gc.collect()
    
    return intersection_basis


# ============================================================================
# EXAMPLE
# ============================================================================

def main():
    """Demonstration of optimized non-commutative algebra system."""
    print("=" * 70)
    print("Optimized Non-Commutative Algebra System (COO Format)")
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
    uw5 = generate_uw_basis(5, verbose=True)
    for i, elem in enumerate(uw5):
        print(f"  [{i}] {elem}")
    print()
    
    # Intersection
    print("=" * 70)
    print("Intersection Computation:")
    print("=" * 70)
    print()
    
    uw33 = generate_uw_basis(33, verbose=True)
    uw32 = generate_uw_basis(32, verbose=True)
    intersection = compute_intersection(uw33, g22, uw32, g23, verbose=True)
    
    print("\nIntersection basis:")
    if intersection:
        for i, elem in enumerate(intersection):
            print(f"  [{i}] {elem}")
    else:
        print("  (trivial - only zero)")
    print()


if __name__ == "__main__":
    main()