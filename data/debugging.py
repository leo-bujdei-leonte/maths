"""
Debug script to understand what's in bases[3] and what linear combinations we're generating
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from math import gcd
from fractions import Fraction
from typing import Dict, Iterator, List, Optional, Tuple, Union
from itertools import combinations as itertools_combinations
import random

# [Include all your polynomial code here...]
# (Paste the full Polynomial class, reduce_monomial, etc.)

def generate_linear_combinations_debug(basis: List[Polynomial], max_coeff: int = 10, 
                                       num_terms: int = 2, num_combinations: int = 20) -> List[Polynomial]:
    """Generate random linear combinations with debug output."""
    combinations = []
    
    print(f"\nDEBUG: generate_linear_combinations called with:")
    print(f"  basis size: {len(basis)}")
    print(f"  num_terms: {num_terms}")
    print(f"  num_combinations per index set: {num_combinations}")
    print(f"  max_coeff: {max_coeff}")
    print()
    
    # Show what's in the basis
    print("Basis elements:")
    for i, elem in enumerate(basis):
        print(f"  basis[{i}] = {elem}")
    print()
    
    # Generate combinations of num_terms basis elements
    index_sets = list(itertools_combinations(range(len(basis)), num_terms))
    print(f"Number of index sets: {len(index_sets)}")
    print()
    
    for indices in index_sets:
        print(f"Index set {indices}:")
        for attempt in range(num_combinations):
            coeffs = [random.randint(-max_coeff, max_coeff) for _ in indices]
            
            if all(c == 0 for c in coeffs):
                print(f"  Attempt {attempt}: skipped (all coeffs zero)")
                continue
            
            combo = Polynomial.zero()
            for idx, coeff in zip(indices, coeffs):
                if coeff != 0:
                    combo = combo + (coeff * basis[idx])
            
            if not combo.is_zero():
                combinations.append(combo)
                print(f"  Attempt {attempt}: {coeffs} → {combo}")
            else:
                print(f"  Attempt {attempt}: resulted in zero polynomial")
        print()
    
    return combinations


def main_debug():
    """Debug version of main"""
    print("=" * 80)
    print("Debugging basis structure and linear combinations")
    print("=" * 80)
    
    # Generate basis for degree 3
    print("\nGenerating basis for degree 3:")
    basis_3 = generate_uw_basis(3, verbose=True)
    print(f"\nBasis 3 has {len(basis_3)} elements:")
    for i, elem in enumerate(basis_3):
        print(f"  basis_3[{i}] = {elem}")
    
    # Set seed for reproducibility
    random.seed(25)
    
    # Generate linear combinations
    print("\n" + "=" * 80)
    print("Generating linear combinations for degree 3:")
    print("=" * 80)
    combos_deg3 = generate_linear_combinations_debug(basis_3, max_coeff=10, 
                                                      num_terms=2, num_combinations=5)
    
    print(f"\n\nGenerated {len(combos_deg3)} combinations total")
    print("\nAll generated combinations:")
    for i, combo in enumerate(combos_deg3):
        print(f"  combo[{i}] = {combo}")


if __name__ == "__main__":
    main_debug()