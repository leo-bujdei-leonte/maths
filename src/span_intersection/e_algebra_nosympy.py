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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from collections import defaultdict
from tqdm import tqdm

Monomial = Tuple[int, ...]  # Ordered tuple of indices representing e_{i1} * e_{i2} * ...
CoeffDict = Dict[Monomial, int]  # Sparse representation: monomial -> coefficient


#Monomial Reduction 
# lru_cache: decorator
#Remembers the results of previous function calls
#If you call the function with the same arguments again, it returns the cached result instantly
#Keeps at most maxsize results in memory
#When full, removes the "least recently used" result
@lru_cache(maxsize=100000000)
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
    
    if verbose:
        print(f"  Generated {len(basis)} basis elements")
    
    return basis


# Vector Rationalization

def rationalize_vector(vec: np.ndarray, max_denominator: int = 1000000) -> List[int]:
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
    non_zero = [abs(x) for x in int_vec if x != 0]
    if non_zero:
        common_gcd = reduce(gcd, non_zero)
        int_vec = [x // common_gcd for x in int_vec]
    
    return int_vec


# Intersection Computation

def compute_intersection(
    basis1: List[Polynomial],
    g1: Polynomial,
    basis2: List[Polynomial],
    g2: Polynomial,
    return_matrix: bool = False,
    tolerance: float = 1e-10,
    verbose: bool = True
) -> Union[List[Polynomial], np.ndarray]:
    """
    Compute the intersection of span{basis1 * g1} and span{basis2 * g2}.
    
    This solves a homogeneous linear system by:
    1. Expanding all products basis[i] * g
    2. Building coefficient matrix (rows = monomials, cols = products)
    3. Finding null space via SVD
    4. Reconstructing intersection elements from null vectors
    
    Args:
        basis1: First basis (list of Polynomials)
        g1: First element to multiply with
        basis2: Second basis (list of Polynomials)
        g2: Second element to multiply with
        return_matrix: If True, return coefficient matrix instead of intersection
        tolerance: Threshold for considering singular values as zero
        verbose: If True, print progress information
        
    Returns:
        List of Polynomials spanning the intersection, or coefficient matrix
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
    
    all_monomials: set[Monomial] = set()
    for expansion in expansions:
        all_monomials.update(expansion.terms.keys())
    
    monomial_list = sorted(all_monomials)

    if verbose:
        print(f"  Found {len(monomial_list)} unique monomials")
        if len(monomial_list) < 50:  # Only print if not too many
            print(f"  Monomials: {monomial_list}")
    
    if not monomial_list or not expansions:
        if return_matrix:
            return np.array([])
        return []
    
    # Step 3: Build coefficient matrix
    if verbose:
        print("  Building coefficient matrix...")
    
    mono_to_idx = {mono: i for i, mono in enumerate(monomial_list)}
    v = []
    r = []
    c = []
    for col, expansion in enumerate(expansions):
        for mono, coeff in expansion.terms.items():
            if coeff != 0:
                v.append(float(coeff))
                r.append(mono_to_idx[mono])
                c.append(col)
    
    coo = coo_matrix((np.array(v), (np.array(r), np.array(c))), shape=(len(monomial_list), len(expansions)))
    
    # Step 4: Compute null space
    if verbose:
        print("  Computing null space via SVD for sparse...")
    
    U, s, Vt = svds(coo, k = min(coo.shape)-1, which="SM")
    null_mask = s < tolerance
    null_space =  Vt.T[:, null_mask]

    
    if verbose:
        print(f"  Matrix shape: {coo.shape}")
        print(f"  Null space dimension: {null_space.shape[1]}")
    
    prod = coo @ null_space

    if not np.allclose(prod, np.zeros(prod.shape), tolerance):
        print(prod)
        print("ERROR: SVD did not compute nullspace correctly!")
        exit(1)

    # Step 5: Reconstruct intersection elements
    intersection_basis = []
    n1 = len(basis1)
    
    for col_idx in range(null_space.shape[1]):
        null_vector = null_space[:, col_idx]
        int_coeffs = rationalize_vector(null_vector)
        
        # Build linear combination from basis1 side
        result = Polynomial.zero()
        for i in range(n1):
            if int_coeffs[i] != 0:
                result = result + (int_coeffs[i] * (basis1[i] * g1))
        
        if not result.is_zero():
            intersection_basis.append(result.expand())
    
    if verbose:
        print(f"  Intersection dimension: {len(intersection_basis)}")
    
    return intersection_basis


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

    uw36 = generate_uw_basis(36)
    uw35 = generate_uw_basis(35)
    intersection = compute_intersection(uw36, g22, uw35, g23, verbose=True)

    print("\nIntersection basis:")
    if intersection:
        for i, elem in enumerate(intersection):
            print(f"  [{i}] {elem}")
    else:
        print("  (trivial - only zero)")
    print()
 
if __name__ == "__main__":
    main()