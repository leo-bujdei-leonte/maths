# Quick Start: intersect_uw_bases

## TL;DR

```python
from e_algebra import UW_basis, g, intersect_uw_bases

# Compute intersection directly
intersection = intersect_uw_bases(
    UW1_basis=UW_basis(2),
    g_1=g(2, 2),
    UW2_basis=UW_basis(3),
    g_2=g(2, 3)
)

# Result: List of intersection basis elements
print(f"Dimension: {len(intersection)}")
for elem in intersection:
    print(elem)
```

## What Changed

**Before**: Function returned `(coefficient_matrix, monomial_keys)`
- Required manual null space computation
- User had to construct basis elements

**After**: Function returns `List[Expr]`
- Automatically solves the linear system
- Returns ready-to-use basis elements
- Coefficients are clean integers

## Function Signature

```python
def intersect_uw_bases(
    UW1_basis: List[Expr],
    g_1: Expr,
    UW2_basis: List[Expr],
    g_2: Expr
) -> List[Expr]
```

## Return Value

- **Empty list** `[]` → Trivial intersection (zero only)
- **Non-empty list** → Basis for the intersection subspace
- All elements are **expanded** and have **integer coefficients**

## Example Output

```
Dimension: 2
e_1*e_2*e_3 - e_2*e_4 - e_2**3 - e_3**2
e_1**2*e_2**2 + e_1**2*e_4 - e_1**3*e_3
```

## Test Files

- `test_intersection_simple.py` - Basic examples
- Run: `python src/span_intersection/test_intersection_simple.py`
