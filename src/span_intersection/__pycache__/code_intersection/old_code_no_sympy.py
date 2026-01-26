"""
Optimized non-commutative algebra implementation for computing intersections
of U(W_+) subspaces with non-multiplication rules.

- Polynomial representation with operator overloading

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
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import svds
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from scipy.linalg import lstsq

Monomial = Tuple[int, ...]  # Ordered tuple of indices representing e_{i1} * e_{i2} * ...
CoeffDict = Dict[Monomial, int]  # Sparse representation: monomial -> coefficient


#Monomial Reduction 
# lru_cache: decorator
#Remembers the results of previous function calls
#If you call the function with the same arguments again, it returns the cached result instantly
#Keeps at most maxsize results in memory
#When full, removes the "least recently used" result
@lru_cache(maxsize=10000)
def reduce_monomial(seq: Monomial) -> Tuple[Tuple[Monomial, int], ...]:
    """
    Reduce a monomial sequence to sorted form using the non-commutative rule:
        e_n * e_m = e_m * e_n + (m-n) * e_{n+m} when n > m
    
    Results are cached indefinitely for maximum speed on repeated computations. 
    MIGHT NOT BE THE BEST OPTION SPACE-WISE
    
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


# Polynomial Class
#provides a decorator and functions for adding special methods
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
    # example of usage: can do Polynomial.constant instead of calling the instance
    
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
        
        # Binary exponentiation done by writing the exponent in binary and then dividing it by 2 (more like multiplying) 
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


# Convenience Functions

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


# Partition Generation

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
    
    print(f"[GEN] Finished generate_uw_basis({n})", flush=True)
    
    return basis


# Vector Rationalization

def rationalize_vector(vec: np.ndarray, max_denominator: int = 1000) -> List[int]:
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
    # non_zero = [abs(x) for x in int_vec if x != 0]
    # if non_zero:
    #     common_gcd = reduce(gcd, non_zero)
    #     int_vec = [x // common_gcd for x in int_vec]
    vec_gcd = reduce(gcd, (abs(x) for x in int_vec if x != 0), 0)
    if vec_gcd > 0:
        int_vec = [x // vec_gcd for x in int_vec]
    return int_vec

def fully_expand(poly: Polynomial) -> Polynomial:

    result: Dict[Monomial,int] = defaultdict(int)

    for mono, coeff in poly.terms.items():
        # reduce_monomial returns a tuple of (monomial, coeff)
        reduced = reduce_monomial(mono)
        for r_mono, r_coeff in reduced:
            result[r_mono] += coeff * r_coeff
    return Polynomial(dict(result))


# Intersection Computation
# FIXED: compute_intersection with proper null space reconstruction

def compute_intersection_debug(
    basis1: List[Polynomial],
    g1: Polynomial,
    basis2: List[Polynomial],
    g2: Polynomial,
    tolerance: float = 1e-10,
    verbose: bool = True
) -> dict:
    """
    Compute the intersection of span{basis1 * g1} and span{basis2 * g2}.
    
    FIXED: Properly reconstructs intersection elements from null space
    
    Returns:
        Dictionary with:
        {
            'intersection': List[Polynomial],
            'matrix': coo_matrix,
            'null_space': np.ndarray,
            'singular_values': np.ndarray,
            'monomial_list': List,
            'basis1_size': int,
            'basis2_size': int
        }
    """
    if verbose:
        print("Computing intersection...")
        print(f"  Basis 1: {len(basis1)} elements")
        print(f"  Basis 2: {len(basis2)} elements")
    
    # Step 1: Compute all expansions
    expansions = []
    
    if verbose:
        print("  Expanding basis1 * g1...")
    for poly in basis1:
        expansions.append((poly * g1).expand())
    
    if verbose:
        print("  Expanding basis2 * g2...")
    for poly in basis2:
        expansions.append((poly * g2).expand())
    
    # Step 2: Collect all monomials
    if verbose:
        print("  Collecting monomials...")
    
    all_monomials = set()
    for expansion in expansions:
        all_monomials.update(expansion.terms.keys())
    
    monomial_list = sorted(all_monomials)

    if verbose:
        print(f"  Found {len(monomial_list)} unique monomials")
    
    if not monomial_list or not expansions:
        return {
            'intersection': [],
            'matrix': None,
            'null_space': None,
            'singular_values': None,
            'monomial_list': monomial_list,
            'basis1_size': len(basis1),
            'basis2_size': len(basis2)
        }
    
    # Step 3: Build coefficient matrix
    if verbose:
        print("  Building coefficient matrix...")
    
    mono_to_idx = {mono: i for i, mono in enumerate(monomial_list)}
    v = []
    r = []
    c = []
    
    n1 = len(basis1)
    
    # Add basis1 * g1 (positive coefficients)
    for col, expansion in enumerate(expansions[:n1]):
        for mono, coeff in expansion.terms.items():
            if coeff != 0:
                v.append(float(coeff))
                r.append(mono_to_idx[mono])
                c.append(col)
    
    # Add basis2 * g2 (negative coefficients)
    for col, expansion in enumerate(expansions[n1:]):
        for mono, coeff in expansion.terms.items():
            if coeff != 0:
                v.append(-float(coeff))
                r.append(mono_to_idx[mono])
                c.append(col + n1)

    print(f"[MAT] Building sparse matrix: rows={len(monomial_list)}, cols={len(expansions)}")
    coo = coo_matrix((np.array(v), (np.array(r), np.array(c))), 
                     shape=(len(monomial_list), len(expansions)))
    
    # Step 4: Compute null space using DENSE SVD (more reliable)
    print("[SVD] Using dense SVD for accurate null space computation...")
    A_dense = coo.toarray()
    
    from scipy.linalg import svd as dense_svd
    U, s, Vt = dense_svd(A_dense, full_matrices=True)
    
    print(f"[SVD] Singular values: min={np.min(s):.2e}, max={np.max(s):.2e}")
    
    null_mask = s < tolerance
    null_space = Vt.T[:, null_mask]
    
    print(f"[SVD] Found {np.sum(null_mask)} null vectors")
    print(f"[SVD] Null space shape: {null_space.shape}")
    
    # Verify null space
    prod = coo @ null_space
    max_residual = np.max(np.abs(prod))
    print(f"[SVD] Verification: max||A @ null_space|| = {max_residual:.2e}")
    
    if max_residual > tolerance:
        print("WARNING: Null space residual is large!")

    # Step 5: Reconstruct intersection elements from null space
    print("[RECON] Reconstructing intersection elements...")
    
    intersection_basis = []
    
    for col_idx in range(null_space.shape[1]):
        null_vector = null_space[:, col_idx]
        
        # Rationalize ALL coefficients
        int_coeffs = rationalize_vector(null_vector)
        
        # The null vector has n1 + n2 components
        n2 = len(basis2)
        
        # Reconstruct from basis1 side: sum_i c1_i * (basis1[i] * g1)
        result_from_basis1 = Polynomial.zero()
        for i in range(n1):
            if int_coeffs[i] != 0:
                result_from_basis1 = result_from_basis1 + (int_coeffs[i] * (basis1[i] * g1))
        
        # Reconstruct from basis2 side: sum_j c2_j * (basis2[j] * g2)
        result_from_basis2 = Polynomial.zero()
        for j in range(n2):
            if int_coeffs[n1 + j] != 0:
                result_from_basis2 = result_from_basis2 + (int_coeffs[n1 + j] * (basis2[j] * g2))
        
        # They should be equal (up to sign)
        result_from_basis1_expanded = result_from_basis1.expand()
        result_from_basis2_expanded = result_from_basis2.expand()
        
        # Debug: check if they're actually equal
        diff = (result_from_basis1_expanded + result_from_basis2_expanded).expand()
        
        if not diff.is_zero():
            print(f"  [WARN] Null vector {col_idx}: basis1 result ≠ -basis2 result")
            print(f"    basis1 result: {result_from_basis1_expanded}")
            print(f"    basis2 result: {result_from_basis2_expanded}")
            print(f"    difference: {diff}")
        
        # Use the basis1 reconstruction
        if not result_from_basis1_expanded.is_zero():
            intersection_basis.append(result_from_basis1_expanded)
        elif not result_from_basis2_expanded.is_zero():
            # If basis1 is zero, use basis2 (shouldn't happen)
            intersection_basis.append(result_from_basis2_expanded)
    
    if verbose:
        print(f"  Intersection dimension: {len(intersection_basis)}")
    
    return {
        'intersection': intersection_basis,
        'matrix': coo,
        'null_space': null_space,
        'singular_values': s,
        'monomial_list': monomial_list,
        'basis1_size': n1,
        'basis2_size': n2
    }


# USAGE:
# result = compute_intersection_fixed(basis1, g1, basis2, g2)
# intersection = result['intersection']
# 
# Then verify with:
# verify_results = quick_verify(
#     intersection,
#     basis1, g1, basis2, g2,
#     matrix_coo=result['matrix'],
#     null_space=result['null_space'],
#     singular_values=result['singular_values']
# )
def verify_null_space_directly(matrix_coo, null_space, tolerance=1e-8):
    """
    Most basic check: Does the null space actually satisfy A @ null_space = 0?
    """
    print("=" * 80)
    print("TEST 1: NULL SPACE VERIFICATION (A @ null_space = 0)")
    print("=" * 80)
    print()
    
    print(f"Matrix shape: {matrix_coo.shape}")
    print(f"Null space shape: {null_space.shape}")
    print()
    
    # Handle empty null space
    if null_space is None or null_space.size == 0:
        print("NULL SPACE IS EMPTY!")
        print()
        print("DIAGNOSIS: No null vectors were extracted from the SVD.")
        print("This means either:")
        print("  1. The SVD didn't find any singular values below tolerance")
        print("  2. The matrix has full rank (no non-trivial null space)")
        print("  3. SVD computed all singular values as non-zero")
        print()
        return False
    
    residual_matrix = matrix_coo @ null_space
    residual_norms = np.linalg.norm(residual_matrix, axis=0)
    
    print(f"Residual norms for each null vector:")
    for i, norm in enumerate(residual_norms):
        status = "Yes" if norm < tolerance else "No"
        print(f"  Vector {i}: ||A @ v{i}|| = {norm:.2e} {status}")
    print()
    
    if len(residual_norms) > 0:
        max_residual = np.max(residual_norms)
        print(f"Maximum residual: {max_residual:.2e}")
        print(f"Tolerance: {tolerance:.2e}")
        print()
        
        if max_residual < tolerance:
            print(" NULL SPACE IS VALID")
            return True
        else:
            print(" NULL SPACE IS INVALID")
            return False
    else:
        print(" NO NULL VECTORS FOUND")
        return False


def verify_intersection_membership(intersection_elements, basis1, g1, basis2, g2, verbose=True):
    """
    Verify that each intersection element can be expressed as:
    - A linear combination of {basis1[i] * g1}
    - AND as a linear combination of {basis2[j] * g2}
    """
    print("=" * 80)
    print("TEST 2: INTERSECTION MEMBERSHIP VERIFICATION")
    print("=" * 80)
    print()
    
    if not intersection_elements:
        print("WARNING: No intersection elements to verify")
        return False
    
    from scipy.linalg import lstsq
    
    failures = []
    all_monomials = set()
    
    # Expand all basis elements
    expansions1 = [(poly * g1).expand() for poly in basis1]
    expansions2 = [(poly * g2).expand() for poly in basis2]
    
    for exp in expansions1 + expansions2:
        all_monomials.update(exp.terms.keys())
    
    print(f"Total unique monomials in span: {len(all_monomials)}")
    print()
    
    # For each intersection element
    for elem_idx, elem in enumerate(intersection_elements):
        print(f"Element {elem_idx}: {elem}")
        print(f"  Degree: {elem.degree()}, Terms: {elem.num_terms()}")
        print()
        
        mono_list = sorted(all_monomials)
        mono_to_idx = {m: i for i, m in enumerate(mono_list)}
        
        # Check membership in span{basis1 * g1}
        print(f"  Checking membership in span{{basis1 * g1}}...")
        
        mat1 = np.zeros((len(mono_list), len(basis1)))
        for col, exp in enumerate(expansions1):
            for mono, coeff in exp.terms.items():
                row = mono_to_idx[mono]
                mat1[row, col] = coeff
        
        vec_elem = np.zeros(len(mono_list))
        for mono, coeff in elem.terms.items():
            if mono in mono_to_idx:
                row = mono_to_idx[mono]
                vec_elem[row] = coeff
        
        try:
            x1, residuals1, rank1, s1 = lstsq(mat1, vec_elem)
            residual1 = np.linalg.norm(mat1 @ x1 - vec_elem)
            
            print(f"    Residual: {residual1:.2e}, Rank: {rank1}/{len(basis1)}")
            
            if residual1 < 1e-6:
                print(f"    elem in span{{basis1 * g1}}")
            else:
                print(f"     elem NOT in span{{basis1 * g1}}")
                failures.append((elem_idx, 1))
        except Exception as e:
            print(f"    Error: {e}")
            failures.append((elem_idx, 1))
        
        print()
        
        # Check membership in span{basis2 * g2}
        print(f"  Checking membership in span{{basis2 * g2}}...")
        
        mat2 = np.zeros((len(mono_list), len(basis2)))
        for col, exp in enumerate(expansions2):
            for mono, coeff in exp.terms.items():
                row = mono_to_idx[mono]
                mat2[row, col] = coeff
        
        try:
            x2, residuals2, rank2, s2 = lstsq(mat2, vec_elem)
            residual2 = np.linalg.norm(mat2 @ x2 - vec_elem)
            
            print(f"    Residual: {residual2:.2e}, Rank: {rank2}/{len(basis2)}")
            
            if residual2 < 1e-6:
                print(f"     elem in span{{basis2 * g2}}")
            else:
                print(f"    elem NOT in span{{basis2 * g2}}")
                failures.append((elem_idx, 2))
        except Exception as e:
            print(f"    Error: {e}")
            failures.append((elem_idx, 2))
        
        print()
    
    if failures:
        print("=" * 80)
        print(f" VERIFICATION FAILED: {len(failures)} elements not in intersection")
        return False
    else:
        print("=" * 80)
        print(" ALL ELEMENTS ARE VALID MEMBERS OF BOTH SPACES")
        return True


def check_intersection_consistency(intersection_elements, verbose=True):
    """
    Check that computing with intersection elements gives consistent results.
    """
    print("=" * 80)
    print("TEST 3: INTERSECTION ELEMENT CONSISTENCY")
    print("=" * 80)
    print()
    
    if not intersection_elements:
        return True
    
    all_consistent = True
    
    for idx, elem in enumerate(intersection_elements):
        print(f"Element {idx}")
        
        # Expand multiple times - should be idempotent
        expanded1 = elem.expand()
        expanded2 = elem.expand()
        
        if expanded1.terms != expanded2.terms:
            print("   expand() is non-deterministic!")
            all_consistent = False
        else:
            print("   expand() is consistent")
        
        # Multiple expands
        e1 = elem.expand()
        e2 = e1.expand()
        e3 = e2.expand()
        
        if e1.terms != e2.terms or e2.terms != e3.terms:
            print("   Multiple expands give different results!")
            all_consistent = False
        else:
            print("   Multiple expands are consistent")
        
        # Polynomial arithmetic consistency
        sum1 = elem + elem
        sum2 = elem + elem
        
        if sum1.terms != sum2.terms:
            print("   Addition is non-deterministic!")
            all_consistent = False
        else:
            print("  ✓ Addition is consistent")
        
        print()
    
    if all_consistent:
        print(" All elements are internally consistent")
    else:
        print(" Some elements show inconsistent behavior")
    
    print()
    return all_consistent


def diagnose_svd_output(matrix_coo, singular_values, tolerance=1e-10):
    """
    Diagnose what's happening in the SVD.
    """
    print("=" * 80)
    print("SVD DIAGNOSIS")
    print("=" * 80)
    print()
    
    print(f"Matrix shape: {matrix_coo.shape}")
    print(f"Matrix rank: at most {min(matrix_coo.shape)}")
    print()
    
    print(f"Singular values from svds(): {singular_values}")
    print(f"Number of singular values: {len(singular_values)}")
    print()
    
    print(f"Tolerance for null space: {tolerance}")
    print()
    
    null_mask = singular_values < tolerance
    num_null = np.sum(null_mask)
    
    print(f"Singular values < tolerance: {num_null}")
    if num_null > 0:
        print(f"  Values: {singular_values[null_mask]}")
    print()
    
    print(f"Singular values >= tolerance: {len(singular_values) - num_null}")
    if len(singular_values) - num_null > 0:
        print(f"  Values: {singular_values[~null_mask]}")
    print()
    
    print("ANALYSIS:")
    print(f"  Smallest singular value: {np.min(singular_values):.2e}")
    print(f"  Largest singular value: {np.max(singular_values):.2e}")
    print(f"  Condition number: {np.max(singular_values) / np.min(singular_values):.2e}")
    print()
    
    if num_null == 0:
        print("  No singular values below tolerance!")
        print()
        print("Possible causes:")
        print("  1. You requested k=min(shape)-1, missing the smallest singular value")
        print("  2. The matrix actually has full rank")
        print("  3. Your tolerance is too strict")
        print()
        print("RECOMMENDATION:")
        print("  - Try using dense SVD instead of sparse")
        print("  - Or try larger tolerance (e.g., 1e-8 instead of 1e-10)")
    else:
        print(f" Found {num_null} vector(s) in null space")
    
    print()


    """
    Run all 3 verification tests.
    """
    print()
    print("=" * 80)
    print("=" + "INTERSECTION VERIFICATION SUITE".center(78) + "=")
    print("=" * 80)
    print()
    
    results = {}
    
    # Test 1
    if matrix_coo is not None and null_space is not None:
        results['null_space'] = verify_null_space_directly(matrix_coo, null_space)
    else:
        print("Skipping null space test (matrices not provided)")
        results['null_space'] = None
    
    print()
    
    # Test 2
    results['membership'] = verify_intersection_membership(
        intersection_elements, basis1, g1, basis2, g2
    )
    
    print()
    
    # Test 3
    results['consistency'] = check_intersection_consistency(intersection_elements)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if results['null_space'] is not None:
        print(f"Null space valid: {' YES' if results['null_space'] else ' NO'}")
    
    print(f"Membership valid: {' YES' if results['membership'] else ' NO'}")
    print(f"Consistency valid: {'YES' if results['consistency'] else 'NO'}")
    print()
    
    overall = all(v for v in results.values() if v is not None)
    if overall:
        print("INTERSECTION RESULT IS VALID ")
    else:
        print("INTERSECTION RESULT IS INVALID")
    
    print()
    print("=" * 80)
    print()
    
    return results

# COMPLETE VERIFICATION MODULE
# Copy this ENTIRE file into your project as "verification.py"
# Then: from verification import quick_verify


def diagnose_svd_output(matrix_coo, singular_values, tolerance=1e-10):
    """
    Diagnose what's happening in the SVD.
    """
    print("=" * 80)
    print("SVD DIAGNOSIS")
    print("=" * 80)
    print()
    
    print(f"Matrix shape: {matrix_coo.shape}")
    print(f"Matrix rank: at most {min(matrix_coo.shape)}")
    print()
    
    print(f"Singular values from svds(): {singular_values}")
    print(f"Number of singular values: {len(singular_values)}")
    print()
    
    print(f"Tolerance for null space: {tolerance}")
    print()
    
    null_mask = singular_values < tolerance
    num_null = np.sum(null_mask)
    
    print(f"Singular values < tolerance: {num_null}")
    if num_null > 0:
        print(f"  Values: {singular_values[null_mask]}")
    print()
    
    print(f"Singular values >= tolerance: {len(singular_values) - num_null}")
    if len(singular_values) - num_null > 0:
        print(f"  Values: {singular_values[~null_mask]}")
    print()
    
    print("ANALYSIS:")
    print(f"  Smallest singular value: {np.min(singular_values):.2e}")
    print(f"  Largest singular value: {np.max(singular_values):.2e}")
    print(f"  Condition number: {np.max(singular_values) / np.min(singular_values):.2e}")
    print()
    
    if num_null == 0:
        print(" No singular values below tolerance!")
        print()
        print("Possible causes:")
        print("  1. You requested k=min(shape)-1, missing the smallest singular value")
        print("  2. The matrix actually has full rank")
        print("  3. Your tolerance is too strict")
        print()
        print("RECOMMENDATION:")
        print("  - Try using dense SVD instead of sparse")
        print("  - Or try larger tolerance (e.g., 1e-8 instead of 1e-10)")
    else:
        print(f"✓ Found {num_null} vector(s) in null space")
    
    print()


def verify_null_space_directly(matrix_coo, null_space, tolerance=1e-8):
    """
    Most basic check: Does the null space actually satisfy A @ null_space = 0?
    """
    print("=" * 80)
    print("TEST 1: NULL SPACE VERIFICATION (A @ null_space = 0)")
    print("=" * 80)
    print()
    
    print(f"Matrix shape: {matrix_coo.shape}")
    print(f"Null space shape: {null_space.shape}")
    print()
    
    # Handle empty null space
    if null_space is None or null_space.size == 0:
        print(" NULL SPACE IS EMPTY!")
        print()
        print("DIAGNOSIS: No null vectors were extracted from the SVD.")
        print("This means either:")
        print("  1. The SVD didn't find any singular values below tolerance")
        print("  2. The matrix has full rank (no non-trivial null space)")
        print("  3. SVD computed all singular values as non-zero")
        print()
        return False
    
    residual_matrix = matrix_coo @ null_space
    residual_norms = np.linalg.norm(residual_matrix, axis=0)
    
    print(f"Residual norms for each null vector:")
    for i, norm in enumerate(residual_norms):
        status = "Yes" if norm < tolerance else "No"
        print(f"  Vector {i}: ||A @ v{i}|| = {norm:.2e} {status}")
    print()
    
    if len(residual_norms) > 0:
        max_residual = np.max(residual_norms)
        print(f"Maximum residual: {max_residual:.2e}")
        print(f"Tolerance: {tolerance:.2e}")
        print()
        
        if max_residual < tolerance:
            print(" NULL SPACE IS VALID")
            return True
        else:
            print(" NULL SPACE IS INVALID")
            return False
    else:
        print(" NO NULL VECTORS FOUND")
        return False


def verify_intersection_membership(intersection_elements, basis1, g1, basis2, g2, verbose=True):
    """
    Verify that each intersection element can be expressed as:
    - A linear combination of {basis1[i] * g1}
    - AND as a linear combination of {basis2[j] * g2}
    """
    print("=" * 80)
    print("TEST 2: INTERSECTION MEMBERSHIP VERIFICATION")
    print("=" * 80)
    print()
    
    if not intersection_elements:
        print("WARNING: No intersection elements to verify")
        return False
    
    failures = []
    all_monomials = set()
    
    # Expand all basis elements
    expansions1 = [(poly * g1).expand() for poly in basis1]
    expansions2 = [(poly * g2).expand() for poly in basis2]
    
    for exp in expansions1 + expansions2:
        all_monomials.update(exp.terms.keys())
    
    print(f"Total unique monomials in span: {len(all_monomials)}")
    print()
    
    # For each intersection element
    for elem_idx, elem in enumerate(intersection_elements):
        print(f"Element {elem_idx}: {elem}")
        print(f"  Degree: {elem.degree()}, Terms: {elem.num_terms()}")
        print()
        
        mono_list = sorted(all_monomials)
        mono_to_idx = {m: i for i, m in enumerate(mono_list)}
        
        # Check membership in span{basis1 * g1}
        print(f"  Checking membership in span{{basis1 * g1}}...")
        
        mat1 = np.zeros((len(mono_list), len(basis1)))
        for col, exp in enumerate(expansions1):
            for mono, coeff in exp.terms.items():
                row = mono_to_idx[mono]
                mat1[row, col] = coeff
        
        vec_elem = np.zeros(len(mono_list))
        for mono, coeff in elem.terms.items():
            if mono in mono_to_idx:
                row = mono_to_idx[mono]
                vec_elem[row] = coeff
        
        try:
            x1, residuals1, rank1, s1 = lstsq(mat1, vec_elem)
            residual1 = np.linalg.norm(mat1 @ x1 - vec_elem)
            
            print(f"    Residual: {residual1:.2e}, Rank: {rank1}/{len(basis1)}")
            
            if residual1 < 1e-6:
                print(f"     elem in span{{basis1 * g1}}")
            else:
                print(f"     elem NOT in span{{basis1 * g1}}")
                failures.append((elem_idx, 1))
        except Exception as e:
            print(f"    Error: {e}")
            failures.append((elem_idx, 1))
        
        print()
        
        # Check membership in span{basis2 * g2}
        print(f"  Checking membership in span{{basis2 * g2}}...")
        
        mat2 = np.zeros((len(mono_list), len(basis2)))
        for col, exp in enumerate(expansions2):
            for mono, coeff in exp.terms.items():
                row = mono_to_idx[mono]
                mat2[row, col] = coeff
        
        try:
            x2, residuals2, rank2, s2 = lstsq(mat2, vec_elem)
            residual2 = np.linalg.norm(mat2 @ x2 - vec_elem)
            
            print(f"    Residual: {residual2:.2e}, Rank: {rank2}/{len(basis2)}")
            
            if residual2 < 1e-6:
                print(f"     elem in span{{basis2 * g2}}")
            else:
                print(f"     elem NOT in span{{basis2 * g2}}")
                failures.append((elem_idx, 2))
        except Exception as e:
            print(f"    Error: {e}")
            failures.append((elem_idx, 2))
        
        print()
    
    if failures:
        print("=" * 80)
        print(f" VERIFICATION FAILED: {len(failures)} elements not in intersection")
        return False
    else:
        print("=" * 80)
        print(" ALL ELEMENTS ARE VALID MEMBERS OF BOTH SPACES")
        return True


def check_intersection_consistency(intersection_elements, verbose=True):
    """
    Check that computing with intersection elements gives consistent results.
    """
    print("=" * 80)
    print("TEST 3: INTERSECTION ELEMENT CONSISTENCY")
    print("=" * 80)
    print()
    
    if not intersection_elements:
        return True
    
    all_consistent = True
    
    for idx, elem in enumerate(intersection_elements):
        print(f"Element {idx}")
        
        # Expand multiple times - should be idempotent
        expanded1 = elem.expand()
        expanded2 = elem.expand()
        
        if expanded1.terms != expanded2.terms:
            print("  expand() is non-deterministic!")
            all_consistent = False
        else:
            print("  expand() is consistent")
        
        # Multiple expands
        e1 = elem.expand()
        e2 = e1.expand()
        e3 = e2.expand()
        
        if e1.terms != e2.terms or e2.terms != e3.terms:
            print("   Multiple expands give different results!")
            all_consistent = False
        else:
            print("   Multiple expands are consistent")
        
        # Polynomial arithmetic consistency
        sum1 = elem + elem
        sum2 = elem + elem
        
        if sum1.terms != sum2.terms:
            print("  Addition is non-deterministic!")
            all_consistent = False
        else:
            print("  Addition is consistent")
        
        print()
    
    if all_consistent:
        print("All elements are internally consistent")
    else:
        print(" Some elements show inconsistent behavior")
    
    print()
    return all_consistent


def quick_verify(intersection_elements, basis1, g1, basis2, g2, matrix_coo=None, null_space=None, singular_values=None):
    """
    Run all 3 verification tests.
    
    Args:
        intersection_elements: List of Polynomial results
        basis1, g1, basis2, g2: Original parameters
        matrix_coo: The sparse matrix (optional)
        null_space: The null space array (optional)
        singular_values: Singular values from SVD (optional)
    
    Returns:
        Dictionary with test results
    """
    print()
    print("" * 80)
    print("" + "INTERSECTION VERIFICATION SUITE".center(78) + "█")
    print("" * 80)
    print()
    
    results = {}
    
    # Diagnosis
    if matrix_coo is not None and singular_values is not None:
        diagnose_svd_output(matrix_coo, singular_values)
    
    print()
    
    # Test 1
    if matrix_coo is not None and null_space is not None:
        results['null_space'] = verify_null_space_directly(matrix_coo, null_space)
    else:
        print("Skipping null space test (matrices not provided)")
        results['null_space'] = None
    
    print()
    
    # Test 2
    results['membership'] = verify_intersection_membership(
        intersection_elements, basis1, g1, basis2, g2
    )
    
    print()
    
    # Test 3
    results['consistency'] = check_intersection_consistency(intersection_elements)
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if results['null_space'] is not None:
        print(f"Null space valid: {' YES' if results['null_space'] else ' NO'}")
    
    print(f"Membership valid: {' YES' if results['membership'] else ' NO'}")
    print(f"Consistency valid: {' YES' if results['consistency'] else ' NO'}")
    print()
    
    overall = all(v for v in results.values() if v is not None)
    if overall:
        print(" INTERSECTION RESULT IS VALID ")
    else:
        print(" INTERSECTION RESULT IS INVALID")
    
    print()
    print("=" * 80)
    print()
    
    return results
# Example

def main():
    """Demonstration of the non-commutative algebra system."""
    print("=" * 70)
    print("Non-Commutative Algebra System")
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
    
    # Intersection
    print("=" * 70)
    print("Intersection Computation:")
    print("=" * 70)
    print()
    
    #uw36 = generate_uw_basis(11)
    #uw35 = generate_uw_basis(10)
    #intersection = compute_intersection(uw36, g22, uw35, g23, verbose=True)
    #M = compute_intersection(uw6, g22, uw5, g23,return_matrix=True)
    #print(M)

    #uw35 = generate_uw_basis(5)
    #uw36 = generate_uw_basis(6)
    
    #intersection = compute_intersection(uw35,  e(1)**4, uw36,e(1)**3 , verbose=True)



    #print(intersection)

    #intersection_2 = compute_intersection([e(1),e(1)**2], 3*e(1)*e(1)+3*e(2), [1], -6*e(2)**2+6*e(3)*e(1)+e(1)**4, verbose=True)

    #a = fully_expand(e(3)*e(1)**3)
    #print(a)
    #print(intersection_2)

    """
    for target_degree in range(17, 18):  # Start from 2 since we need target_degree - 1
        # Compute intersection of bases[target_degree-1] * poly4 with bases[target_degree] * poly3
        intersection = compute_intersection(
            generate_uw_basis(target_degree), e(5),
            generate_uw_basis(target_degree + 1), e(4),
            verbose=False
        )
        print(f"For each degree {target_degree+5}, the intersection is {intersection}")
 

    for target_degree in range(17, 18):  # Start from 2 since we need target_degree - 1
        # Compute intersection of bases[target_degree-1] * poly4 with bases[target_degree] * poly3
        intersection = compute_intersection(
            generate_uw_basis(target_degree+1), e(4),
            generate_uw_basis(target_degree), e(5),
            verbose=False
        )
        print(f"For each degree {target_degree+5}, the intersection is {intersection}")

    
    for target_degree in range(1, 35):  # Start from 2 since we need target_degree - 1
        # Compute intersection of bases[target_degree-1] * poly4 with bases[target_degree] * poly3
        intersection = compute_intersection(
            generate_uw_basis(target_degree), 2 * e(1) * e(3) - 5 * e(2) ** 2,
            generate_uw_basis(target_degree + 1), 7 * e(1) * e(2) + 9 * e(3),
            verbose=False
        )
        print(f"For each degree {target_degree}, the intersection is {intersection}")
        """

if __name__ == "__main__":
    main()

    result = compute_intersection_debug(
        generate_uw_basis(18), e(4),
        generate_uw_basis(17), e(5),
        verbose=True
    )
    
    # Extract the results
    intersection = result['intersection']
    matrix_coo = result['matrix']
    null_space = result['null_space']
    singular_values = result['singular_values']
    
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    # Run verification
    verify_results = quick_verify(
        intersection,
        generate_uw_basis(18), e(4),
        generate_uw_basis(17), e(5),
        matrix_coo=matrix_coo,
        null_space=null_space,
        singular_values=singular_values
    )
    
    print("Verification complete!")