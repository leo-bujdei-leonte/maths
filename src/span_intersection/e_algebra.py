"""
This programe implements the intersection of $U(W_+)_n g(2,2)$ with $U(W_+)_m g(2,3)$
by finding and expanding the basis of the graded pieces of $U(W_+)$ and then impletementing 
a noncommutative relationshop between the basis elements. 


Asymmetric algebra for e_i elements using SymPy with multiplication rule:
e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n

Results are normalized so that products have indices in increasing order. 
(TODO decreasing order as wel)

Note: When m = n (same index), the rule doesn't apply, so e_n * e_n remains
as e_n**2 or e_n*e_n. Powers of the same element don't simplify further.
"""

from typing import Tuple, List, Any, Iterator
from sympy import Expr, Add, Mul, Pow, S
from sympy.core.expr import AtomicExpr
import numpy as np
from tqdm import tqdm as tqdm

class E(AtomicExpr):
    """
    Non-commutative basis element e_i with custom multiplication rule.

    Multiplication rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n.
    """

    is_commutative: bool = False
    index: int

    def __new__(cls, index: int) -> "E":
        # instantiation
        obj: E = AtomicExpr.__new__(cls)
        obj.index = int(index)
        return obj

    def __eq__(self, other: Any) -> bool:
        # equality check
        if isinstance(other, E):
            return self.index == other.index
        return False

    def _hashable_content(self) -> Tuple[int]:
        # sympy hash
        return (self.index,)
    
    def __hash__(self) -> int:
        # dict hash
        return hash((self.__class__.__name__, self.index))

    def _latex(self, _printer: Any) -> str:
        # for printing
        return f"e_{{{self.index}}}"

    def __repr__(self) -> str:
        # for printing
        return f"e_{self.index}"

    def _sympystr(self, _printer: Any) -> str:
        # for printing
        return f"e_{self.index}"

    __str__ = __repr__


def multiply_e_elements(expr: Expr) -> Expr:
    """
    Recursively apply the multiplication rule and expand the products until all elements are sorted.

    Rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when n > m

    Args:
        expr: SymPy expression containing E elements (could be single E, a product, a sum etc).

    Returns:
        Normalized expression with E elements in increasing order.
    """
    if isinstance(expr, E):
        return expr

    if isinstance(expr, Add):
        return Add(*[multiply_e_elements(arg) for arg in expr.args])

    if isinstance(expr, Pow):
        # For powers, just return them as-is
        # (e_1 * e_1 stays as e_1*e_1 or e_1**2 since indices are equal)
        return expr

    if isinstance(expr, Mul):
        # Peform noncommutative multiplication at the first out-of-order index

        coeff: Expr = S.One # scalar coefficient of the multiplication
        e_elements: List[E] = [] # es to be multiplied together

        # fill in e_elements
        for arg in expr.args:
            if isinstance(arg, E):
                e_elements.append(arg)
            elif isinstance(arg, Mul):
                # Handle nested Mul - shouldn't happen
                # return multiply_e_elements(expr.expand())
                raise NotImplementedError()
            elif isinstance(arg, Add):
                # Need to distribute
                return multiply_e_elements(expr.expand())
            elif isinstance(arg, Pow) and isinstance(arg.base, E):
                # Expand power of E: e_i**n = e_i * e_i * ... (n times)
                if arg.exp.is_Integer and arg.exp > 0:
                    for _ in range(int(arg.exp)):
                        e_elements.append(arg.base)
                else:
                    raise NotImplementedError("Non-integer or negative powers of E are not supported")
            else: # scalar
                coeff *= arg

        # If no E elements or only one, return as is
        if len(e_elements) == 0:
            return coeff
        if len(e_elements) == 1:
            return coeff * e_elements[0]

        # Check if E elements are in increasing order
        indices: List[int] = [e.index for e in e_elements]
        if indices == sorted(indices):
            #returns an expression
            return coeff * Mul(*e_elements, evaluate=False)

        # Find first out-of-order pair
        swap_idx: int | None = None
        for i in range(len(e_elements) - 1):
            if e_elements[i].index > e_elements[i + 1].index:
                swap_idx = i
                break

        assert swap_idx is not None, "swap_idx should have been found"

        # Apply commutation rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m)
        # where n = e_elements[i].index, m = e_elements[i+1].index, and n > m
        n: int = e_elements[swap_idx].index
        m: int = e_elements[swap_idx + 1].index

        # Build the two terms
        left_part: List[E] = e_elements[:swap_idx]
        right_part: List[E] = e_elements[swap_idx + 2:]

        # First term: swapped elements
        swapped: List[E] = left_part + [e_elements[swap_idx + 1], e_elements[swap_idx]] + right_part
        term1: Expr
        if swapped:
            term1 = coeff * Mul(*swapped, evaluate=False)
        else:
            term1 = coeff

        # Second term: correction (m - n) * e_(n+m)
        correction_coeff: int = m - n
        corrected: List[E] = left_part + [E(n + m)] + right_part
        term2: Expr
        if corrected:
            term2 = coeff * correction_coeff * Mul(*corrected, evaluate=False)
        else:
            term2 = coeff * correction_coeff * E(n + m)

        # Recursively process both terms
        result: Expr = multiply_e_elements(term1) + multiply_e_elements(term2)
        return result

    # unrecognised expression type
    raise NotImplementedError


def e(n: int) -> E:
    """
    Create convenience constructor for 'E'.

    Args:
        n: Index of the basis element

    Returns:
        Basis element e_n
    """
    return E(n)


def expand_e(expr: Expr) -> Expr:
    """
    Expand powers of 'E' elements into explicit products using the noncommutative
    rules.

    Args:
        expr: Expression to expand

    Returns:
        Expanded and normalized expression
    """
    # First expand using SymPy's expand
    expanded: Expr = expr.expand()

    # Then apply our custom multiplication rules
    result: Expr = multiply_e_elements(expanded)

    # Expand again to collect terms
    return result.expand() if hasattr(result, 'expand') else result


def g(n: int, m: int) -> Expr:
    """
    Construct the elements g(n, m) defined as:
    g(n, m) = e(n) * e(m) - e(1) * e(n+m-1) + e(n+m) * (n-1)

    Use it later in the computation of the intersection.
    Args:
        n: First index
        m: Second index

    Returns:
        Expanded expression for g(n, m)
    """
    return expand_e(e(n) * e(m) - e(1) * e(n + m - 1) + (e(n + m) * (n - 1)))


def partitions(n: int, I: int = 1) -> Iterator[Tuple[int, ...]]:
    """
    Generate integer partitions of n with parts in increasing order.(TODO decreasing)

    Args:
        n: The integer to partition
        I: Minimum part size (default 1), used for recursion

    Yields:
        Tuples representing partitions in increasing order

    Example:
        >>> list(partitions(3))
        [(3,), (1, 2), (1, 1, 1)]

    Source: https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    """
    yield (n,)
    for i in range(I, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p


def UW_basis(n: int) -> List[Expr]:
    """
    Generate the UW basis for a given total index n.

    The PBW  U(W_+)_n increasing basis consists of normalized products $e_{i_1}*...*e_{i_k}$ such that 
    $i_{1} + ... + i_{k} = n$.

    Args:
        n: Total sum of indices

    Returns:
        List of normalized basis expressions in reverse partition order

    Example:
        >>> UW_basis(3)
        [e_3, e_1*e_2, e_1**3]
    """
    res: List[Expr] = []
    for p in  tqdm(list(partitions(n))[::-1], desc=f"Constructing UW_{n}-basis"):
        # Create product of e elements for this partition
        product: Expr = S.One
        for idx in p:
            product = product * e(idx)
        res.append(expand_e(product))
    return res


def intersect_uw_bases(
    UW1_basis: List[Expr],
    g_1: Expr,
    UW2_basis: List[Expr],
    g_2: Expr,
    return_matrix: bool = False
) -> List[Expr]:
    """
    Compute the intersection of the two U(W_+) subspaces multiplies by their respective g elements.
    

    Solves the homogeneous linear system to find elements that can be expressed
    in both forms:
        sum(a_i * (UW1_basis[i] * g_1)) = sum(b_j * (UW2_basis[j] * g_2))

    This is equivalent to solving:
        sum(a_i * (UW1_basis[i] * g_1)) + sum(b_j * (UW2_basis[j] * g_2)) = 0

    This is done by building a coefficeint matrix in a shared monomial basis. 
    Args:
        UW1_basis: First UW basis (list of expressions)
        g_1: First g element
        UW2_basis: Second UW basis (list of expressions)
        g_2: Second g element

    Returns:
        List of intersection basis elements (may be empty if intersection is trivial)
    """

    # Compute all expansions
    expansions: List[Expr] = []

    # Add UW1_basis[i] * g_1 for all i
    for basis_elem in UW1_basis:
        expansion = expand_e(basis_elem * g_1)
        expansions.append(expansion)

    # Add UW2_basis[j] * g_2 for all j
    for basis_elem in UW2_basis:
        expansion = expand_e(basis_elem * g_2)  # TODO maybe this needs to be minus (but the linalg is the same)
        expansions.append(expansion)

    # Extract all unique monomials (products of E elements) across all expansions
    all_monomials: set[Tuple[int, ...]] = set()

    for expansion in tqdm(expansions,desc = "Expansions"):
        monomials = _extract_monomials(expansion)
        all_monomials.update(monomials.keys())

    # Sort monomials for consistent ordering
    monomial_keys = sorted(all_monomials)

    # --- PRINT THE MONOMIALS ---
    print("\nMONOMIALS USED AS ROWS:")
    for m in monomial_keys:
        print(m)
    print("Total monomials:", len(monomial_keys))
    
    # Build coefficient matrix
    # Rows: unique monomials
    # Columns: expansions e_k g_ij
    coefficient_matrix: List[List[int]] = []
    for i, monomial_key in tqdm(enumerate(monomial_keys), desc = "Constructing matrix"):
        row=[]
        for j,expansion in enumerate(expansions):
            monomials = _extract_monomials(expansion) # TODO can be saved from the previous loop
            coeff = monomials.get(monomial_key, 0) # same as monomials[monomial_key] if monomial_key in monomials else 0
            row.append(coeff)
        coefficient_matrix.append(row)    
    if return_matrix:
        return coefficient_matrix

    # Convert to numpy array and find null space
    C = np.array(coefficient_matrix, dtype=float)

    # Use SVD to find null space
    if C.size == 0:
        return []

    _U, s, Vt = np.linalg.svd(C, full_matrices=True)
    rank = np.sum(s > 1e-10)
    null_space = Vt[rank:].T

    # Build intersection basis elements from null space vectors
    intersection_basis: List[Expr] = []

    for i in range(null_space.shape[1]):
        vec = null_space[:, i]

        # Round to clean up numerical errors and find rational approximations
        # Use a tolerance to determine which coefficients are effectively zero
        vec_cleaned = _rationalize_vector(vec)

        # Construct intersection element from first part (UW1_basis coefficients)
        intersection_elem: Expr | None = None

        for j in range(len(UW1_basis)):
            coeff = vec_cleaned[j]
            if coeff != 0:
                term = coeff * (UW1_basis[j] * g_1)

                if intersection_elem is None:
                    intersection_elem = term
                else:
                    intersection_elem = intersection_elem + term

        if intersection_elem is not None:
            intersection_elem = expand_e(intersection_elem)
            intersection_basis.append(intersection_elem)

    return intersection_basis


def _rationalize_vector(vec: np.ndarray, max_denom: int = 1000) -> List[int]:
    """
    Convert a float vector to rational approximations, then scale to integers.

    Args:
        vec: Input vector with float values
        max_denom: Maximum denominator for rational approximation

    Returns:
        List of integer values (scaled and GCD-reduced)
    """
    from fractions import Fraction
    from math import gcd
    from functools import reduce

    # Convert to fractions
    fracs: List[Fraction] = []
    for val in vec:
        if abs(val) < 1e-10:
            fracs.append(Fraction(0))
        else:
            fracs.append(Fraction(val).limit_denominator(max_denom))

    # Find LCM of all denominators
    denominators = [f.denominator for f in fracs if f != 0]
    if not denominators:
        return [0] * len(vec)

    lcm = reduce(lambda a, b: a * b // gcd(a, b), denominators, 1)

    # Scale to integers
    int_vec = [int(f * lcm) for f in fracs]

    # Reduce by GCD
    vec_gcd = reduce(gcd, (abs(x) for x in int_vec if x != 0), 0)
    if vec_gcd > 0:
        int_vec = [x // vec_gcd for x in int_vec]

    return int_vec


def _extract_monomials(expr: Expr) -> dict[Tuple[int, ...], int]:
    """
    Extract monomials and their coefficients from an expression.

    A monomial is represented as a sorted tuple of E element indices.
    For example, e_1*e_2*e_3 is represented as (1, 2, 3).
    Powers like e_2**2 are represented as (2, 2).

    Args:
        expr: SymPy expression containing E elements

    Returns:
        Dictionary mapping monomial keys to their integer coefficients
    """
    from collections import defaultdict

    monomials: dict[Tuple[int, ...], int] = defaultdict(int)

    # Handle the case where expr is already expanded
    if isinstance(expr, Add):
        # Sum of terms
        for term in expr.args:
            monomial_key, coeff = _extract_single_term(term)
            monomials[monomial_key] += coeff
    else:
        # Single term
        monomial_key, coeff = _extract_single_term(expr)
        monomials[monomial_key] += coeff

    # Convert defaultdict to regular dict and remove zero coefficients (dictionary comprehension)
    return {k: v for k, v in monomials.items() if v != 0}


def _extract_single_term(term: Expr) -> Tuple[Tuple[int, ...], int]:
    """
    Extract monomial key and coefficient from a single term.

    Args:
        term: A single term (could be E, Mul, Pow, or numeric)

    Returns:
        Tuple of (monomial_key, coefficient) where monomial_key is a sorted
        tuple of indices
    """
    if isinstance(term, E):
        # Single E element: e_i
        return ((term.index,), 1)

    elif isinstance(term, Mul):
        # Product: coefficient * e_i * e_j * ...
        coeff: int = 1
        indices: List[int] = []

        for arg in term.args:
            if isinstance(arg, E):
                indices.append(arg.index)
            elif isinstance(arg, Pow) and isinstance(arg.base, E):
                # Handle e_i**n
                exp_val = int(arg.exp) if arg.exp.is_Integer else 0
                indices.extend([arg.base.index] * exp_val)
            elif arg.is_Number:
                coeff *= int(arg)
            else:
                # Nested structure - shouldn't happen with normalized expressions
                pass

        monomial_key = tuple(sorted(indices)) # we will hash these - so we want them to map to the same key
        return (monomial_key, coeff)

    elif isinstance(term, Pow) and isinstance(term.base, E):
        # Power: e_i**n
        exp_val = int(term.exp) if term.exp.is_Integer else 1
        indices = [term.base.index] * exp_val
        monomial_key = tuple(sorted(indices))
        return (monomial_key, 1)

    elif term.is_Number:
        # Just a constant (shouldn't typically appear alone)
        return ((), int(term))

    else:
        # Fallback for other cases
        return ((), 0)




# Example usage and tests
if __name__ == "__main__":
    print("Rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when n > m")
    print()

    #print("g(2, 2):")
    g_22 = g(2, 2)
    #print(g_22)
    #print()

    #print("g(2, 3):")
    g_23 = g(2, 3)
    #print(g_23)
    #print()

    # Test UW_basis function
    print("=" * 60)
    print("Testing UW_basis function:")
    print("=" * 60)
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

    # Test interaction between g and UW_basis
    #print("e(1) * e(1) * e(1) * g(2, 2):")
    #result = expand_e(e(1) * e(1) * e(1) * g_22)
    #print(result)
    #print()

    #print("e(3) * g(2, 2):")
    #result = expand_e(e(3) * g_22)
    #print(result)
    #print()

    #print("e(1) * e(1) * g(2, 3):")
    #result = expand_e(e(1) * e(1) * g_23)
    #print(result)
    #print()

    #print("e(2) * g(2, 3):")
    #result = expand_e(e(2) * g_23)
    #print(result)
    #print()

    # Test intersect_uw_bases function
    print("=" * 60)
    print("Testing intersect_uw_bases function:")
    print("=" * 60)
    print()

    print("Example: Intersection of UW_basis(5)*g(2,2) and UW_basis(4)*g(2,3)")
    print()



    print("Example: Intersection of UW_basis(6)*g(2,2) and UW_basis(5)*g(2,3)")
    print()
    uw_6 = UW_basis(6)
    uw_5 = UW_basis(5)

    # # Show the bases
    print("UW_basis(6):", uw_6)
    print("g(2,2):", g_22)
    print()
    print("UW_basis(5):", uw_5)
    print("g(2,3):", g_23)
    print()

    #Show expansions for context
    print("Expansions of UW_basis(6)[i] * g(2,2):")
    for i, basis_elem in enumerate(uw_6):
        expansion = expand_e(basis_elem * g_22)
        print(f"  [{i}]: {expansion}")
    print()

    print("Expansions of UW_basis(5)[i] * g(2,3):")
    for i, basis_elem in enumerate(uw_5):
        expansion = expand_e(g_22 * basis_elem)
        print(f"  [{i}]: {expansion}")
    print()
 

    #intersection30 = intersect_uw_bases(UW_basis(26), g_22, UW_basis(25), g_23)
    #print(f"Dimension of intersection30: {len(intersection30)}")

    print("Sparcity coefficient")
#print("n\aapproximate_p(n-4) + approximate_p(n-5) - approximate_p(n))\approximate_p(n-4) + approximate_p(n-5) - approximate_p(n)-(approximate_p(n-5) + approximate_p(n-6) - approximate_p(n-1)")
print("-" * 50)

#for n in range(1, 35):

UW1 = UW_basis(6)
UW2 = UW_basis(5)

    # get the coefficient matrix instead of the intersection
M = intersect_uw_bases(
    UW1, g(2,2),
    UW2, g(2,3),
    return_matrix= True)

M = np.array(M)
zeros=np.count_nonzero(M == 0)
total = M.size
sparsity = zeros / total

print(f"{10}\t{M.shape}\t{M.size}\t{zeros}\t{sparsity}")
