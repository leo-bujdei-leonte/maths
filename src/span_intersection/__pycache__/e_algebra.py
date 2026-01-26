"""
This program implements the intersection of U(W_+)_n g(2,2) with U(W_+)_m g(2,3)
by finding and expanding the basis of the graded pieces of U(W_+) and then implementing 
a noncommutative relationship between the basis elements.

Asymmetric algebra for e_i elements using SymPy with multiplication rule:
e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n

Results are normalized so that products have indices in increasing order.
"""

from typing import Tuple, List, Any, Iterator, Dict
from sympy import Expr, Add, Mul, Pow, S, Matrix
from sympy.core.expr import AtomicExpr
import numpy as np
from tqdm import tqdm

class E(AtomicExpr):
    """
    Non-commutative basis element e_i with custom multiplication rule.

    Multiplication rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n.
    """

    is_commutative: bool = False
    index: int

    def __new__(cls, index: int) -> "E":
        obj: E = AtomicExpr.__new__(cls)
        obj.index = int(index)
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, E):
            return self.index == other.index
        return False

    def _hashable_content(self) -> Tuple[int]:
        return (self.index,)
    
    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.index))

    def _latex(self, _printer: Any) -> str:
        return f"e_{{{self.index}}}"

    def __repr__(self) -> str:
        return f"e_{self.index}"

    def _sympystr(self, _printer: Any) -> str:
        return f"e_{self.index}"

    __str__ = __repr__


def multiply_e_elements(expr: Expr) -> Expr:
    """
    Recursively apply the multiplication rule and expand products until all elements are sorted.

    Rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when n > m
    """
    if isinstance(expr, E):
        return expr

    if isinstance(expr, Add):
        return Add(*[multiply_e_elements(arg) for arg in expr.args])

    if isinstance(expr, Pow):
        return expr

    if isinstance(expr, Mul):
        coeff: Expr = S.One
        e_elements: List[E] = []

        for arg in expr.args:
            if isinstance(arg, E):
                e_elements.append(arg)
            elif isinstance(arg, Add):
                return multiply_e_elements(expr.expand())
            elif isinstance(arg, Pow) and isinstance(arg.base, E):
                if arg.exp.is_Integer and arg.exp > 0:
                    for _ in range(int(arg.exp)):
                        e_elements.append(arg.base)
                else:
                    raise NotImplementedError("Non-integer or negative powers of E are not supported")
            else:
                coeff *= arg

        if len(e_elements) == 0:
            return coeff
        if len(e_elements) == 1:
            return coeff * e_elements[0]

        indices: List[int] = [e.index for e in e_elements]
        if indices == sorted(indices):
            return coeff * Mul(*e_elements, evaluate=False)

        swap_idx: int | None = None
        for i in range(len(e_elements) - 1):
            if e_elements[i].index > e_elements[i + 1].index:
                swap_idx = i
                break

        assert swap_idx is not None, "swap_idx should have been found"

        n: int = e_elements[swap_idx].index
        m: int = e_elements[swap_idx + 1].index

        left_part: List[E] = e_elements[:swap_idx]
        right_part: List[E] = e_elements[swap_idx + 2:]

        swapped: List[E] = left_part + [e_elements[swap_idx + 1], e_elements[swap_idx]] + right_part
        term1: Expr
        if swapped:
            term1 = coeff * Mul(*swapped, evaluate=False)
        else:
            term1 = coeff

        correction_coeff: int = m - n
        corrected: List[E] = left_part + [E(n + m)] + right_part
        term2: Expr
        if corrected:
            term2 = coeff * correction_coeff * Mul(*corrected, evaluate=False)
        else:
            term2 = coeff * correction_coeff * E(n + m)

        result: Expr = multiply_e_elements(term1) + multiply_e_elements(term2)
        return result

    raise NotImplementedError


def e(n: int) -> E:
    """Create convenience constructor for 'E'."""
    return E(n)


def expand_e(expr: Expr) -> Expr:
    """
    Expand powers of 'E' elements into explicit products using noncommutative rules.
    """
    expanded: Expr = expr.expand()
    result: Expr = multiply_e_elements(expanded)
    return result.expand() if hasattr(result, 'expand') else result


def g(n: int, m: int) -> Expr:
    """
    Construct the elements g(n, m) defined as:
    g(n, m) = e(n) * e(m) - e(1) * e(n+m-1) + e(n+m) * (n-1)
    """
    return expand_e(e(n) * e(m) - e(1) * e(n + m - 1) + (e(n + m) * (n - 1)))


def partitions(n: int, I: int = 1) -> Iterator[Tuple[int, ...]]:
    """
    Generate integer partitions of n with parts in increasing order.
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p


def UW_basis(n: int) -> List[Expr]:
    """
    Generate the UW basis for a given total index n.

    The PBW U(W_+)_n increasing basis consists of normalized products e_{i_1}*...*e_{i_k} 
    such that i_{1} + ... + i_{k} = n.
    """
    res: List[Expr] = []
    for p in tqdm(list(partitions(n))[::-1], desc=f"Constructing UW_{n}-basis"):
        product: Expr = S.One
        for idx in p:
            product = product * e(idx)
        res.append(expand_e(product))
    return res


def _extract_single_term(term: Expr) -> Tuple[Tuple[int, ...], int]:
    """
    Extract monomial key and coefficient from a single term.
    Properly handles Mul expressions to collect all E indices.
    """
    if isinstance(term, E):
        return ((term.index,), 1)

    elif isinstance(term, Mul):
        coeff = 1
        indices: List[int] = []

        for arg in term.args:
            if isinstance(arg, E):
                indices.append(arg.index)
            elif isinstance(arg, Pow):
                if isinstance(arg.base, E):
                    if arg.exp.is_Integer and int(arg.exp) > 0:
                        exp_val = int(arg.exp)
                        indices.extend([arg.base.index] * exp_val)
                    else:
                        pass
                else:
                    if arg.exp.is_Integer:
                        coeff *= int(arg.base) ** int(arg.exp) if arg.base.is_Number else 1
            elif arg.is_Number:
                try:
                    coeff *= int(arg)
                except (TypeError, ValueError):
                    coeff *= arg
            elif arg.is_Symbol:
                pass

        # Keep indices in the order they appear (should already be sorted from expand_e)
        monomial_key = tuple(indices) if indices else ()
        return (monomial_key, coeff)

    elif isinstance(term, Pow):
        if isinstance(term.base, E):
            if term.exp.is_Integer and int(term.exp) > 0:
                exp_val = int(term.exp)
                indices = [term.base.index] * exp_val
                monomial_key = tuple(indices)
                return (monomial_key, 1)
        return ((), 0)

    elif term.is_Number:
        try:
            return ((), int(term))
        except (TypeError, ValueError):
            return ((), 1)

    else:
        return ((), 0)


def _extract_monomials(expr: Expr) -> Dict[Tuple[int, ...], int]:
    """
    Extract monomials and their coefficients from an expression.

    A monomial is represented as a tuple of E element indices in order.
    """
    from collections import defaultdict

    monomials: Dict[Tuple[int, ...], int] = defaultdict(int)

    if isinstance(expr, Add):
        for term in expr.args:
            monomial_key, coeff = _extract_single_term(term)
            monomials[monomial_key] += coeff
    else:
        monomial_key, coeff = _extract_single_term(expr)
        monomials[monomial_key] += coeff

    # Remove zero coefficients and ensure all keys are tuples
    result = {}
    for k, v in monomials.items():
        if v != 0:
            if not isinstance(k, tuple):
                k = (k,) if k else ()
            result[k] = v
    return result


def _extract_monomials_with_zeros(expr: Expr) -> Dict[Tuple[int, ...], int]:
    """
    Extract monomials and their coefficients from an expression.
    Includes monomials with zero coefficients.

    A monomial is represented as a tuple of E element indices in order.
    """
    from collections import defaultdict

    monomials: Dict[Tuple[int, ...], int] = defaultdict(int)

    if isinstance(expr, Add):
        for term in expr.args:
            monomial_key, coeff = _extract_single_term(term)
            monomials[monomial_key] += coeff
    else:
        monomial_key, coeff = _extract_single_term(expr)
        monomials[monomial_key] += coeff

    # Keep all monomials including zeros
    result = {}
    for k, v in monomials.items():
        if not isinstance(k, tuple):
            k = (k,) if k else ()
        result[k] = v
    return result


def intersect_uw_bases(
    UW1_basis: List[Expr],
    g_1: Expr,
    UW2_basis: List[Expr],
    g_2: Expr,
    verbose: bool = True,
    return_matrix: bool = False
) -> Tuple[bool, Dict[str, int]]:
    """
    Compute the intersection of two U(W_+) subspaces multiplied by their respective g elements.
    
    Solves the homogeneous linear system:
        sum(a_i * (UW1_basis[i] * g_1)) = sum(b_j * (UW2_basis[j] * g_2))
    """
    expansions: List[Expr] = []

    for basis_elem in UW1_basis:
        expansion = expand_e(basis_elem * g_1)
        expansions.append(expansion)

    for basis_elem in UW2_basis:
        expansion = expand_e(basis_elem * g_2)
        expansions.append(-expansion)

    all_monomials: set = set()
    all_monomials_with_zeros: set = set()

    for expansion in tqdm(expansions, desc="Extracting monomials"):
        monomials = _extract_monomials(expansion)
        all_monomials.update(monomials.keys())
        
        # Also track monomials with zeros
        monomials_with_zeros = _extract_monomials_with_zeros(expansion)
        all_monomials_with_zeros.update(monomials_with_zeros.keys())

    # Use the version with zeros if it's larger (means we're missing some monomials)
    if len(all_monomials_with_zeros) > len(all_monomials):
        all_monomials = all_monomials_with_zeros

    monomial_keys = sorted(all_monomials)

    if verbose:
        print("\nMONOMIALS USED AS ROWS:")
        for m in monomial_keys:
            print(m)
        print(f"Total monomials: {len(monomial_keys)}")
        
        # Check for (2,2,2) specifically
        if (2, 2, 2) in monomial_keys:
            print("\n✓ (2,2,2) is present in monomials!")
        else:
            print("\n✗ (2,2,2) is NOT present in monomials")
    
    coefficient_matrix: List[List[int]] = []
    for i, monomial_key in tqdm(enumerate(monomial_keys), desc="Constructing matrix"):
        row = []
        for j, expansion in enumerate(expansions):
            monomials = _extract_monomials_with_zeros(expansion)
            coeff = monomials.get(monomial_key, 0)
            row.append(coeff)
        coefficient_matrix.append(row)

    if return_matrix:
        return coefficient_matrix

    C = np.array(coefficient_matrix, dtype=float)

    if C.size == 0:
        return False, {"intersection_exists": False, "rank": 0, "nullity": 0}

    if verbose:
        print("\n[STEP 4] Computing rank using exact arithmetic...")
    
    M = Matrix(C)
    rank = M.rank()
    nullity = M.cols - rank
    
    if verbose:
        print(f"  Rank: {rank}")
        print(f"  Nullity (= # columns - rank): {nullity}")
    
    if verbose:
        print("\n[STEP 5] Applying Rank-Nullity Theorem...")
    
    intersection_exists = nullity > 0
    
    if verbose:
        if intersection_exists:
            print(f"  ✓ INTERSECTION EXISTS")
            print(f"    Dimension of intersection: {nullity}")
        else:
            print(f"  ✗ NO NON-TRIVIAL INTERSECTION")
            print(f"    (Only trivial solution: c = d = 0)")
        print("=" * 70)
    
    debug_info = {
        "intersection_exists": intersection_exists,
        "rank": rank,
        "nullity": nullity
    }
    
    return intersection_exists, debug_info


if __name__ == "__main__":
    print("Rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when n > m")
    print()

    g_22 = g(2, 2)
    g_23 = g(2, 3)

    print("=" * 60)
    print("Testing UW_basis function:")
    print("=" * 60)
    print()

    print("Basic Examples:")
    print(f"e(2) * e(1) = {e(1)**3*(e(3)-4*e(1)* e(2)+5*e(1)**3)}")
    print(f"e(3) * e(1) = {e(1)*e(2)* (e(3)-4*e(1)* e(2)+5*e(1)**3)}")
    print(f"e(3) * e(2) * e(1) = {e(3) * (e(3)-4*e(1)* e(2)+5*e(1)**3)}")
    print()
    

    print("UW_basis(2):")
    uw_2 = UW_basis(2)
    for i, basis_elem in enumerate(uw_2):
        print(f"  [{i}]: {basis_elem}")
    print()

    print("UW_basis(3):")
    uw_3 = UW_basis(3)
    for i, basis_elem in enumerate(uw_3):
        print(f"  [{i}]: {basis_elem}")
    print()

    print("=" * 60)
    print("Testing intersect_uw_bases function:")
    print("=" * 60)
    print()

    print("Example: Intersection of UW_basis(6)*g(2,2) and UW_basis(5)*g(2,3)")
    print()
    uw_6 = UW_basis(6)
    uw_5 = UW_basis(5)

    print("UW_basis(6):", uw_6)
    print("g(2,2):", g_22)
    print()
    print("UW_basis(5):", uw_5)
    print("g(2,3):", g_23)
    print()

    print("Expansions of UW_basis(6)[i] * g(2,2):")
    for i, basis_elem in enumerate(uw_6):
        expansion = expand_e(basis_elem * g_22)
        print(f"  [{i}]: {expansion}")
    print()

    print("Expansions of UW_basis(5)[i] * g(2,3):")
    for i, basis_elem in enumerate(uw_5):
        expansion = expand_e(basis_elem * g_23)
        print(f"  [{i}]: {expansion}")
    print()
 
    for target_degree in range(2, 3):
        intersection_exists, debug_info = intersect_uw_bases(
            UW_basis(target_degree + 1), 
            e(3) - 4*e(1)*e(2) + 5*e(1)**3,
            UW_basis(target_degree),
            10*e(4) - 7*e(2)*e(2),
            verbose=True,
            return_matrix=False
        )
        print(f"For degree {target_degree + 4}: exists={intersection_exists}, debug_info={debug_info}")