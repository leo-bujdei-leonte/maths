"""
Asymmetric algebra for e_i elements using SymPy with multiplication rule:
e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n

Results are normalized so that products have indices in increasing order.

Note: When m = n (same index), the rule doesn't apply, so e_n * e_n remains
as e_n**2 or e_n*e_n. Powers of the same element don't simplify further.
"""

from typing import Tuple, List, Any, Iterator, TYPE_CHECKING
from sympy import Expr, Add, Mul, Pow, S
from sympy.core.expr import AtomicExpr
from functools import total_ordering

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@total_ordering
class E(AtomicExpr):
    """
    Non-commutative basis element e_i with custom multiplication rule.

    Multiplication rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n
    """

    is_commutative: bool = False
    index: int

    def __new__(cls, index: int) -> "E":
        obj: E = AtomicExpr.__new__(cls)
        obj.index = int(index)
        return obj

    def _hashable_content(self) -> Tuple[int]:
        return (self.index,)

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, E):
            return self.index < other.index
        return NotImplemented  # type: ignore

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, E):
            return self.index == other.index
        return False

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
    Recursively apply the multiplication rule to normalize products of E elements.

    Rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when n > m

    Args:
        expr: SymPy expression containing E elements

    Returns:
        Normalized expression with E elements in increasing order
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
        # Extract coefficient and E elements
        coeff: Expr = S.One
        e_elements: List[E] = []

        for arg in expr.args:
            if isinstance(arg, E):
                e_elements.append(arg)
            elif isinstance(arg, Mul):
                # Handle nested Mul - shouldn't happen but just in case
                return multiply_e_elements(expr.expand())
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
            else:
                coeff *= arg

        # If no E elements or only one, return as is
        if len(e_elements) == 0:
            return coeff
        if len(e_elements) == 1:
            return coeff * e_elements[0]

        # Check if E elements are in increasing order
        indices: List[int] = [e.index for e in e_elements]
        if indices == sorted(indices):
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

    # For other types, recursively process arguments if any
    if hasattr(expr, 'args') and expr.args:
        new_args: List[Expr] = [multiply_e_elements(arg) for arg in expr.args]
        return expr.func(*new_args)

    return expr


def e(n: int) -> E:
    """
    Create a basis element e_n.

    Args:
        n: Index of the basis element

    Returns:
        Basis element e_n
    """
    return E(n)


def expand_e(expr: Expr, expand_powers: bool = False) -> Expr:
    """
    Expand and normalize an expression involving E elements.

    Args:
        expr: Expression to expand
        expand_powers: If True, expand powers like e_1**2 into e_1*e_1

    Returns:
        Expanded and normalized expression
    """
    # First expand using SymPy's expand
    expanded: Expr = expr.expand()

    # Optionally expand powers before processing
    if expand_powers:
        expanded = _expand_e_powers(expanded)

    # Then apply our custom multiplication rules
    result: Expr = multiply_e_elements(expanded)

    # If expand_powers is True, recursively expand any powers that were created
    if expand_powers:
        max_iterations: int = 10
        for _ in range(max_iterations):
            if not _has_e_powers(result):
                break
            result = _expand_e_powers(result)
            expanded_result: Expr = result.expand() if hasattr(result, 'expand') else result
            result = multiply_e_elements(expanded_result)

    # Expand again to collect terms
    return result.expand() if hasattr(result, 'expand') else result


def _has_e_powers(expr: Expr) -> bool:
    """
    Check if expression contains powers of E elements.

    Args:
        expr: Expression to check

    Returns:
        True if expression contains E elements raised to a power
    """
    if isinstance(expr, Pow) and isinstance(expr.base, E):
        return True
    elif hasattr(expr, 'args'):
        return any(_has_e_powers(arg) for arg in expr.args)
    return False


def _expand_e_powers(expr: Expr) -> Expr:
    """
    Expand powers of E elements into products.

    Args:
        expr: Expression to expand

    Returns:
        Expression with powers expanded
    """
    if isinstance(expr, E):
        return expr
    elif isinstance(expr, Pow) and isinstance(expr.base, E):
        if expr.exp.is_Integer and expr.exp > 0:
            result: Expr = expr.base
            for _ in range(int(expr.exp) - 1):
                result = result * expr.base
            return result
        return expr
    elif hasattr(expr, 'args') and expr.args:
        new_args: List[Expr] = [_expand_e_powers(arg) for arg in expr.args]
        return expr.func(*new_args)
    return expr


def g(n: int, m: int) -> Expr:
    """
    Compute the g(n, m) element defined as:
    g(n, m) = e(n) * e(m) - e(1) * e(n+m-1) + e(n+m) * (n-1)

    Args:
        n: First index
        m: Second index

    Returns:
        Expanded expression for g(n, m)
    """
    return expand_e(e(n) * e(m) - e(1) * e(n + m - 1) + (e(n + m) * (n - 1)))


def partitions(n: int, I: int = 1) -> Iterator[Tuple[int, ...]]:
    """
    Generate integer partitions of n with parts in increasing order.

    Args:
        n: The integer to partition
        I: Minimum part size (default 1)

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

    Returns all products of basis elements e_i where the indices sum to n,
    with each product normalized using the algebra rules.

    Args:
        n: Total sum of indices

    Returns:
        List of normalized basis expressions in reverse partition order

    Example:
        >>> UW_basis(3)
        [e_3, e_1*e_2 - e_3, e_1**3]
    """
    res: List[Expr] = []
    for p in list(partitions(n))[::-1]:
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
    g_2: Expr
) -> List[Expr]:
    """
    Find the intersection of two UW bases times their respective g elements.

    Solves the homogeneous linear system to find elements that can be expressed
    in both forms:
        sum(a_i * (UW1_basis[i] * g_1)) = sum(b_j * (UW2_basis[j] * g_2))

    This is equivalent to solving:
        sum(a_i * (UW1_basis[i] * g_1)) + sum(b_j * (UW2_basis[j] * g_2)) = 0

    Args:
        UW1_basis: First UW basis (list of expressions)
        g_1: First g element
        UW2_basis: Second UW basis (list of expressions)
        g_2: Second g element

    Returns:
        List of intersection basis elements (may be empty if intersection is trivial)
    """
    import numpy as np

    # Compute all expansions
    expansions: List[Expr] = []

    # Add UW1_basis[i] * g_1 for all i
    for basis_elem in UW1_basis:
        expansion = expand_e(basis_elem * g_1)
        expansions.append(expansion)

    # Add UW2_basis[j] * g_2 for all j
    for basis_elem in UW2_basis:
        expansion = expand_e(basis_elem * g_2)
        expansions.append(expansion)

    # Extract all unique monomials (products of E elements) across all expansions
    all_monomials: set[Tuple[int, ...]] = set()

    for expansion in expansions:
        monomials = _extract_monomials(expansion)
        all_monomials.update(monomials.keys())

    # Sort monomials for consistent ordering
    monomial_keys = sorted(all_monomials)

    # Build coefficient matrix
    # Rows: unique monomials
    # Columns: expansions
    coefficient_matrix: List[List[int]] = []

    for monomial_key in monomial_keys:
        row: List[int] = []
        for expansion in expansions:
            monomials = _extract_monomials(expansion)
            coeff = monomials.get(monomial_key, 0)
            row.append(coeff)
        coefficient_matrix.append(row)

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
                term = UW1_basis[j] * g_1
                term = coeff * term

                if intersection_elem is None:
                    intersection_elem = term
                else:
                    intersection_elem = intersection_elem + term

        if intersection_elem is not None:
            intersection_elem = expand_e(intersection_elem)
            intersection_basis.append(intersection_elem)

    return intersection_basis


def _rationalize_vector(vec: "np.ndarray", max_denom: int = 1000) -> List[int]:
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

    # Convert defaultdict to regular dict and remove zero coefficients
    return {k: v for k, v in monomials.items() if v != 0}


def format_monomial(monomial_key: Tuple[int, ...]) -> str:
    """
    Format a monomial key as a readable string.

    Args:
        monomial_key: Tuple of sorted indices representing a monomial

    Returns:
        String representation like "e_1*e_2*e_3" or "1" for empty tuple

    Example:
        >>> format_monomial((1, 2, 3))
        'e_1*e_2*e_3'
        >>> format_monomial((2, 2))
        'e_2**2'
        >>> format_monomial(())
        '1'
    """
    if not monomial_key:
        return "1"

    # Group consecutive identical indices to use power notation
    result = []
    i = 0
    while i < len(monomial_key):
        idx = monomial_key[i]
        count = 1
        while i + count < len(monomial_key) and monomial_key[i + count] == idx:
            count += 1

        if count == 1:
            result.append(f"e_{idx}")
        else:
            result.append(f"e_{idx}**{count}")

        i += count

    return "*".join(result)


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

        monomial_key = tuple(sorted(indices))
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

    # Test basic commutation
    print("e(2) * e(1):")
    result = expand_e(e(2) * e(1))
    print(result)
    print("Expected: e_1*e_2 - e_3")
    print()

    print("e(3) * e(1):")
    result = expand_e(e(3) * e(1))
    print(result)
    print("Expected: e_1*e_3 - 2*e_4")
    print()

    print("e(3) * e(2):")
    result = expand_e(e(3) * e(2))
    print(result)
    print("Expected: e_2*e_3 - e_5")
    print()

    # Test more complex products
    print("e(1) * e(3) * e(2):")
    result = expand_e(e(1) * e(3) * e(2))
    print(result)
    print()

    print("e(3) * e(2) * e(1):")
    result = expand_e(e(3) * e(2) * e(1))
    print(result)
    print()

    # Test addition
    print("e(1) + 2*e(2) + 3*e(1):")
    result = expand_e(e(1) + 2*e(2) + 3*e(1))
    print(result)
    print()

    # Test product expansion
    print("(e(1) + e(2)) * (e(2) + e(3)):")
    result = expand_e((e(1) + e(2)) * (e(2) + e(3)))
    print(result)
    print()

    # Test triple product
    print("e(1) * e(1) * e(1):")
    result = expand_e(e(1) * e(1) * e(1))
    print(result)
    print()

    # Powers of same element (no simplification since indices are equal)
    print("e(2) * e(2):")
    result = expand_e(e(2) * e(2))
    print(result)
    print("(Stays as e_2**2 since rule only applies when m > n)")
    print()

    # More complex example
    print("(e(1) + e(2)) * e(3):")
    result = expand_e((e(1) + e(2)) * e(3))
    print(result)
    print()

    print("e(4) * e(3) * e(2) * e(1):")
    result = expand_e(e(4) * e(3) * e(2) * e(1))
    print(result)
    print()

    # Test g(n, m) function
    print("=" * 60)
    print("Testing g(n, m) function:")
    print("=" * 60)
    print()

    print("g(2, 2):")
    g_22 = g(2, 2)
    print(g_22)
    print()

    print("g(2, 3):")
    g_23 = g(2, 3)
    print(g_23)
    print()

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
    print("e(1) * e(1) * e(1) * g(2, 2):")
    result = expand_e(e(1) * e(1) * e(1) * g_22)
    print(result)
    print()

    print("e(3) * g(2, 2):")
    result = expand_e(e(3) * g_22)
    print(result)
    print()

    print("e(1) * e(1) * g(2, 3):")
    result = expand_e(e(1) * e(1) * g_23)
    print(result)
    print()

    print("e(2) * g(2, 3):")
    result = expand_e(e(2) * g_23)
    print(result)
    print()

    # Test intersect_uw_bases function
    print("=" * 60)
    print("Testing intersect_uw_bases function:")
    print("=" * 60)
    print()

    print("Example: Intersection of UW_basis(2)*g(2,2) and UW_basis(3)*g(2,3)")
    print()

    # Show the bases
    print("UW_basis(2):", uw_2)
    print("g(2,2):", g_22)
    print()
    print("UW_basis(3):", uw_3)
    print("g(2,3):", g_23)
    print()

    # Show expansions for context
    print("Expansions of UW_basis(2)[i] * g(2,2):")
    for i, basis_elem in enumerate(uw_2):
        expansion = expand_e(basis_elem * g_22)
        print(f"  [{i}]: {expansion}")
    print()

    print("Expansions of UW_basis(3)[j] * g(2,3):")
    for j, basis_elem in enumerate(uw_3):
        expansion = expand_e(basis_elem * g_23)
        print(f"  [{j}]: {expansion}")
    print()

    # Compute intersection
    intersection = intersect_uw_bases(uw_2, g_22, uw_3, g_23)

    print(f"Dimension of intersection: {len(intersection)}")
    print()

    if intersection:
        print("Intersection basis elements:")
        for i, elem in enumerate(intersection):
            print(f"  [{i}]: {elem}")
    else:
        print("Intersection is trivial (contains only zero).")
    print()

    # Test another intersection that might be non-trivial
    print("=" * 60)
    print("Another example: UW_basis(3)*g(2,2) and UW_basis(2)*g(3,2)")
    print("=" * 60)
    print()

    g_32 = g(3, 2)
    print(f"g(3,2) = {g_32}")
    print()

    intersection2 = intersect_uw_bases(uw_3, g_22, uw_2, g_32)
    print(f"Dimension of intersection: {len(intersection2)}")

    if intersection2:
        print("Intersection basis elements:")
        for i, elem in enumerate(intersection2):
            print(f"  [{i}]: {elem}")
    else:
        print("Intersection is trivial (contains only zero).")
    print()



    print("=" * 60)
    print("Another example: UW_basis(5)*g(2,2) and UW_basis(4)*g(2,3)")
    print("=" * 60)
    print()

    intersection3 = intersect_uw_bases(UW_basis(5), g_22, UW_basis(4), g_23)
    print(f"Dimension of intersection: {len(intersection2)}")

    if intersection2:
        print("Intersection basis elements:")
        for i, elem in enumerate(intersection2):
            print(f"  [{i}]: {elem}")
    else:
        print("Intersection is trivial (contains only zero).")
    print()
