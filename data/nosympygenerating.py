"""
Search for pairs of linear combinations at degrees 4 and 3 that produce nontrivial intersections
with U(W_+) bases of all degrees 1-33.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from math import gcd
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Tuple, Union
from itertools import combinations as itertools_combinations
import random
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from tqdm import tqdm

Monomial = Tuple[int, ...]
CoeffDict = Dict[Monomial, int]


@lru_cache(maxsize=10000)
def reduce_monomial(seq: Monomial) -> Tuple[Tuple[Monomial, int], ...]:
    """Reduce a monomial sequence to sorted form using the non-commutative rule."""
    if len(seq) <= 1:
        return ((seq, 1),) if seq else (((), 1),)
    
    if all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)):
        return ((seq, 1),)
    
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
            
            result: Dict[Monomial, int] = {}
            for mono, coeff in term1_pairs:
                result[mono] = result.get(mono, 0) + coeff
            for mono, coeff in term2_pairs:
                result[mono] = result.get(mono, 0) + correction_coeff * coeff
            
            return tuple((m, c) for m, c in result.items() if c != 0)
    
    return ((seq, 1),)


def reduce_to_dict(seq: Monomial) -> CoeffDict:
    """Convert cached tuple result to dictionary."""
    return {mono: coeff for mono, coeff in reduce_monomial(seq)}


@dataclass
class Polynomial:
    """Represents a polynomial as a sparse linear combination of monomials."""
    terms: CoeffDict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure all terms have non-zero coefficients."""
        self.terms = {m: c for m, c in self.terms.items() if c != 0}
    
    @classmethod
    def zero(cls) -> 'Polynomial':
        return cls({})
    
    @classmethod
    def constant(cls, value: int) -> 'Polynomial':
        return cls({(): value}) if value != 0 else cls.zero()
    
    @classmethod
    def from_monomial(cls, monomial: Monomial, coeff: int = 1) -> 'Polynomial':
        if coeff == 0:
            return cls.zero()
        return cls({tuple(monomial): coeff})
    
    @classmethod
    def basis_element(cls, index: int) -> 'Polynomial':
        return cls.from_monomial((index,), 1)
    
    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        result = dict(self.terms)
        for mono, coeff in other.terms.items():
            result[mono] = result.get(mono, 0) + coeff
        return Polynomial(result)
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
    
    def __neg__(self) -> 'Polynomial':
        return Polynomial({m: -c for m, c in self.terms.items()})
    
    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        return self + (-other)
    
    def __mul__(self, other: Union['Polynomial', int]) -> 'Polynomial':
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
    
    def __pow__(self, exponent: int) -> 'Polynomial':
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Only non-negative integer powers supported")
        
        if exponent == 0:
            return Polynomial.constant(1)
        if exponent == 1:
            return Polynomial(dict(self.terms))
        
        result = Polynomial.constant(1)
        base = Polynomial(dict(self.terms))
        exp = exponent
        
        while exp > 0:
            if exp & 1:
                result = result * base
            base = base * base
            exp >>= 1
        
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
    
    def copy(self) -> 'Polynomial':
        return Polynomial(dict(self.terms))
    
    def expand(self) -> 'Polynomial':
        return self.copy()


def e(index: int) -> Polynomial:
    """Create basis element e_i."""
    return Polynomial.basis_element(index)


def integer_partitions(n: int, min_part: int = 1) -> Iterator[Tuple[int, ...]]:
    """Generate all integer partitions of n with parts in increasing order."""
    yield (n,)
    for i in range(min_part, n // 2 + 1):
        for partition in integer_partitions(n - i, i):
            yield (i,) + partition


def generate_uw_basis(n: int, verbose: bool = False) -> List[Polynomial]:
    """Generate the PBW basis for U(W_+)_n."""
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
        print(f"[GEN] Finished generate_uw_basis({n}) with {len(basis)} elements", flush=True)
    
    return basis


def rationalize_vector(vec: np.ndarray, tolerance: float = 1e-10) -> List[int]:
    """
    Convert a floating-point null space vector to integer coefficients.
    Preserves the linear relationships in the null space.
    """
    # Clean up very small values (numerical noise)
    cleaned = vec.copy()
    cleaned[np.abs(cleaned) < tolerance] = 0
    
    if np.allclose(cleaned, 0):
        return [0] * len(vec)
    
    # Find the smallest non-zero magnitude to use as scaling reference
    nonzero_mask = np.abs(cleaned) > tolerance
    if not nonzero_mask.any():
        return [0] * len(vec)
    
    # Try to find integer coefficients by scaling
    # Start with a reasonable multiplier and adjust
    for scale in range(1, 10001):
        scaled = cleaned * scale
        rounded = np.round(scaled)
        
        # Check if rounding preserves the relationships
        if np.allclose(scaled, rounded, atol=1e-6):
            int_vec = [int(r) for r in rounded]
            
            # Reduce by GCD
            vec_gcd = reduce(gcd, (abs(x) for x in int_vec if x != 0), 0)
            if vec_gcd > 0:
                int_vec = [x // vec_gcd for x in int_vec]
            
            return int_vec
    
    # Fallback: use limit_denominator on the full vector (not normalized)
    fractions = [Fraction(v).limit_denominator(1000) for v in cleaned]
    denominators = [f.denominator for f in fractions if f != 0]
    
    if not denominators:
        return [0] * len(vec)
    
    lcm = reduce(lambda a, b: a * b // gcd(a, b), denominators, 1)
    int_vec = [int(f * lcm) for f in fractions]
    
    vec_gcd = reduce(gcd, (abs(x) for x in int_vec if x != 0), 0)
    if vec_gcd > 0:
        int_vec = [x // vec_gcd for x in int_vec]
    
    return int_vec


def normalize_polynomial(poly: Polynomial) -> Polynomial:
    """Normalize a polynomial by dividing all coefficients by their GCD."""
    if poly.is_zero():
        return poly
    
    coeffs = [abs(c) for c in poly.terms.values() if c != 0]
    if not coeffs:
        return poly
    
    common_gcd = reduce(gcd, coeffs)
    if common_gcd > 1:
        return Polynomial({mono: coeff // common_gcd for mono, coeff in poly.terms.items()})
    return poly


def polynomial_to_canonical_form(poly: Polynomial) -> str:
    """Convert polynomial to a canonical string form for deduplication."""
    normalized = normalize_polynomial(poly)
    return str(normalized)


def compute_intersection(
    basis1: List[Polynomial],
    poly1: Polynomial,
    basis2: List[Polynomial],
    poly2: Polynomial,
    tolerance: float = 1e-10,
    verbose: bool = False
) -> Optional[List[Polynomial]]:
    """
    Compute the intersection of span{basis1 * poly1} and span{basis2 * poly2}.
    Returns None if intersection is trivial, otherwise returns list of basis elements.
    
    The null space of the combined matrix represents coefficients (a_1, ..., a_n, b_1, ..., b_m)
    where: a_1*(basis1[0]*poly1) + ... = b_1*(basis2[0]*poly2) + ...

    """
    expansions = []
    for b in basis1:
        expansions.append((b * poly1).expand())
    for b in basis2:
        expansions.append((b * poly2).expand())
    
    all_monomials: set = set()
    for expansion in expansions:
        all_monomials.update(expansion.terms.keys())
    
    monomial_list = sorted(all_monomials)
    
    if not monomial_list or not expansions:
        return None
    
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
    
    coo = coo_matrix((np.array(v), (np.array(r), np.array(c))), 
                     shape=(len(monomial_list), len(expansions)))
    
    try:
        k = min(coo.shape) - 1
        if k <= 0:
            return None
        U, s, Vt = svds(coo, k=k, which="SM")
    except Exception as e:
        return None
    
    null_mask = s < tolerance
    if null_mask.sum() == 0:
        return None
    
    null_space = Vt.T[:, null_mask]
    
    intersection_basis = []
    n1 = len(basis1)
    
    for col_idx in range(null_space.shape[1]):
        null_vector = null_space[:, col_idx]
        int_coeffs = rationalize_vector(null_vector)
        
        result = Polynomial.zero()
        for i in range(n1):
            if int_coeffs[i] != 0:
                result = result + (int_coeffs[i] * (basis1[i] * poly1))
        
        if not result.is_zero():
            intersection_basis.append(result.expand())
    
    return intersection_basis if intersection_basis else None


def generate_linear_combinations(basis: List[Polynomial], max_coeff: int = 10, 
                                 num_terms: int = 4, num_combinations: int = 20) -> List[Polynomial]:
    """Generate random linear combinations of basis elements with small coefficients.
    
    Args:
        basis: Basis elements to combine
        max_coeff: Maximum absolute coefficient value
        num_terms: Number of basis elements to combine in each combination
        num_combinations: Number of random combinations to generate per index set
    """
    combinations = list(basis)  # Don't include basis elements themselves to generated better  linear combinations
    
    # Generate combinations of num_terms basis elements
    # with random coefficients from -max_coeff to +max_coeff
    for indices in itertools_combinations(range(len(basis)), num_terms):
        # Generate multiple random combinations for this index set
        for _ in range(num_combinations):
            coeffs = [random.randint(-max_coeff, max_coeff) for _ in indices]
            
            # Skip if all coefficients are zero
            if all(c == 0 for c in coeffs):
                continue
            
            
            # Build linear combination: sum of coeff * basis[idx]
            combo = Polynomial.zero()
            for idx, coeff in zip(indices, coeffs):
                if coeff != 0:
                    combo = combo + (coeff * basis[idx])
            
            if not combo.is_zero():
                combinations.append(combo)
    
    return combinations


def main():
    """Generate basis elements and search for linear combinations at degrees 4 and 3
    that have nontrivial intersections with all U(W_+) bases from degrees 1-33."""
    print("=" * 80)
    print("Generating U(W_+) basis elements for degrees 1-33")
    print("=" * 80)
    print()
    
    # Generate all bases
    bases = {}
    for degree in tqdm(range(1, 33), desc="Generating bases"):
        bases[degree] = generate_uw_basis(degree, verbose=False)
    
    print(f"\nGenerated bases for degrees 1-9")
    print(f"Basis size for degree 4: {len(bases[4])}")
    print(f"Basis size for degree 3: {len(bases[3])}")
    print()
    
    print("=" * 80)
    print("Generating linear combinations at degrees 4 and 3")
    print("=" * 80)
    print()
    
    # Set random seed for reproducibility
    random.seed(25)
    
    # Generate linear combinations
    print("Generating linear combinations for degree 4...")
    combos_deg4 = generate_linear_combinations(bases[4], max_coeff=10, num_terms=2, num_combinations=10)
    print(f"Generated {len(combos_deg4)} combinations at degree 4")
    
    print("Generating linear combinations for degree 3...")
    combos_deg3 = generate_linear_combinations(bases[3], max_coeff=10, num_terms=2, num_combinations=20)
    print(f"Generated {len(combos_deg3)} combinations at degree 3")
    print()
    
    print("=" * 80)
    print("Searching for linear combination pairs")
    print("with nontrivial intersections across all degrees 2-9")
    print("=" * 80)
    print()
    
    results = []
    
    print(f"Checking {len(combos_deg4)} combinations at degree 4 × {len(combos_deg3)} combinations at degree 3")
    print(f"Testing against 5 U(W_+) bases (degrees 1-5)")
    print()
    
    """
    for i, poly4 in enumerate(tqdm(combos_deg4, desc="Degree 4 combinations")):
        for j, poly3 in enumerate(combos_deg3):
            # For each pair (poly4, poly3), check intersection with all bases
            for target_degree in range(1, 6):  # Start from 2 since we need target_degree - 1
                # Compute intersection of bases[target_degree-1] * poly4 with bases[target_degree] * poly3
                intersection = compute_intersection(
                    bases[target_degree], 5*e(1)**4+2*e(1)*e(3)+9*e(2)**2,
                    bases[target_degree+1], - 3*e(1)**3+5*e(1)*e(2)- 7*e(3),
                    verbose=False
                )
                
                if intersection is not None:
                    # Keep all intersection elements - they are linearly independent from SVD
                    results.append({
                        'poly4': poly4,
                        'poly3': poly3,
                        'target_degree': target_degree,
                        'intersection_dim': len(intersection),
                        'intersection': intersection
                    })
                    """


    # Search through all pairs of linear combinations at degrees 4 and 3
     
    for i, poly4 in enumerate(tqdm(combos_deg4, desc="Degree 4 combinations")):
        for j, poly3 in enumerate(combos_deg3):
            # For each pair (poly4, poly3), check intersection with all bases
            for target_degree in range(1, 15):  # Start from 2 since we need target_degree - 1
                # Compute intersection of bases[target_degree-1] * poly4 with bases[target_degree] * poly3
                intersection = compute_intersection(
                    bases[target_degree], e(4),
                    bases[target_degree+1],poly3,
                    verbose=False
                )
                
                if intersection is not None:
                    # Keep all intersection elements - they are linearly independent from SVD
                    results.append({
                      'poly4': poly4,
                       'poly3': poly3,
                       'target_degree': target_degree,
                        'intersection_dim': len(intersection),
                        'intersection': intersection
                    })
    
    
    """for target_degree in range(1, 33):  # Start from 2 since we need target_degree - 1
        # Compute intersection of bases[target_degree-1] * poly4 with bases[target_degree] * poly3
        intersection = compute_intersection(
            bases[target_degree], e(4),
            bases[target_degree + 1], poly3,
            verbose=False
        )
        print(intersection)"""

    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total nontrivial intersections found: {len(results)}")
    print()
    
    if results:
        # Sort by intersection dimension (largest first)
        results.sort(key=lambda x: x['intersection_dim'], reverse=True)
        
        # Group by polynomial pair
        grouped = defaultdict(list)
        for result in results:
            key_str = (str(result['poly4']), str(result['poly3']))
            grouped[key_str].append(result)
        
        print(f"Found nontrivial intersections in {len(grouped)} linear combination pairs\n")
        
        # Display top results
        pair_list = sorted(grouped.items(), 
                          key=lambda x: max(r['intersection_dim'] for r in x[1]), 
                          reverse=True)[:15]
        
        for idx, (pair_key, res_list) in enumerate(pair_list, 1):
            print(f"[{idx}] Linear Combination Pair:")
            print(f"  Poly4 = {res_list[0]['poly4']}")
            print(f"  Poly3 = {res_list[0]['poly3']}")
            print(f"  Nontrivial intersections with {len(res_list)} degree(s):")
            
            for result in sorted(res_list, key=lambda x: x['intersection_dim'], reverse=True):
                print(f"    Degree {result['target_degree']+4}: dimension {result['intersection_dim']}")
                # Print intersection elements - normalize only for display
                for k, elem in enumerate(result['intersection']):
                    normalized = normalize_polynomial(elem)
                    print(f"      Element {k+1}: {normalized}")
            print()
    else:
        print("No nontrivial intersections found.")


if __name__ == "__main__":
    main()