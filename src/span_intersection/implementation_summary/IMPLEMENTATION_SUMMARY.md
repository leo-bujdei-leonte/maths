# Implementation Summary: intersect_uw_bases

## Overview

I've successfully implemented a complete system for computing span intersections in the asymmetric e_i algebra with full type hints and documentation.

## Functions Implemented

### 1. Core Function: `intersect_uw_bases`

```python
def intersect_uw_bases(
    UW1_basis: List[Expr],
    g_1: Expr,
    UW2_basis: List[Expr],
    g_2: Expr
) -> Tuple[List[List[int]], List[Tuple[int, ...]]]
```

**Purpose**: Extracts the coefficient matrix for solving the intersection of two UW basis spaces.

**Algorithm**:
1. Computes expansions: `UW1_basis[i] * g_1` and `UW2_basis[j] * g_2`
2. Extracts all unique monomials across all expansions
3. Builds coefficient matrix C where:
   - **C[j, i]** = coefficient of monomial j in expansion i
   - Rows = unique monomials (sorted tuples of indices)
   - Columns = expansions (first UW1, then UW2)

**Returns**:
- `coefficient_matrix`: Integer matrix of coefficients
- `monomial_keys`: Ordered list of monomials as tuples

### 2. Helper Functions

#### `_extract_monomials(expr: Expr) -> dict[Tuple[int, ...], int]`
- Extracts all monomials and their coefficients from an expression
- Returns dict mapping monomial keys to integer coefficients

#### `_extract_single_term(term: Expr) -> Tuple[Tuple[int, ...], int]`
- Extracts monomial key and coefficient from a single term
- Handles E, Mul, Pow, and numeric terms

#### `format_monomial(monomial_key: Tuple[int, ...]) -> str`
- Formats monomial tuples as readable strings
- Uses power notation: `(2, 2)` → `"e_2**2"`
- Example: `(1, 2, 3)` → `"e_1*e_2*e_3"`

## Matrix Structure

The coefficient matrix has the following structure:

```
         Column 0    Column 1    ...    Column m    Column m+1    ...
         UW1[0]*g1   UW1[1]*g1   ...   UW1[m-1]*g1  UW2[0]*g2    ...
       ┌─────────────────────────────────────────────────────────────┐
Row 0  │   c_{0,0}     c_{0,1}   ...    c_{0,m-1}    c_{0,m}     ...│
Row 1  │   c_{1,0}     c_{1,1}   ...    c_{1,m-1}    c_{1,m}     ...│
  ...  │     ...         ...     ...       ...          ...       ...│
       └─────────────────────────────────────────────────────────────┘
```

Where:
- **c_{j,i}** = coefficient of the j-th unique monomial in the i-th expansion
- First `len(UW1_basis)` columns correspond to `UW1_basis[i] * g_1`
- Next `len(UW2_basis)` columns correspond to `UW2_basis[j] * g_2`

## Solving for the Intersection

To find the intersection basis:

```python
import numpy as np

# Get the coefficient matrix
matrix, monomials = intersect_uw_bases(UW1_basis, g_1, UW2_basis, g_2)
C = np.array(matrix, dtype=float)

# Find null space of C
U, s, Vt = np.linalg.svd(C)
rank = np.sum(s > 1e-10)
null_space = Vt[rank:].T

# Each column of null_space is a solution vector
# For solution vector x = [a_0, ..., a_{m-1}, b_0, ..., b_{n-1}]:
# Intersection element = sum(a_i * UW1_basis[i] * g_1)
#                      = -sum(b_j * UW2_basis[j] * g_2)
```

## Monomial Representation

Monomials are represented as **sorted tuples of indices**:

| Expression    | Monomial Key | Formatted String |
|--------------|--------------|------------------|
| `e_1`        | `(1,)`       | `"e_1"`          |
| `e_1*e_2`    | `(1, 2)`     | `"e_1*e_2"`      |
| `e_2**2`     | `(2, 2)`     | `"e_2**2"`       |
| `e_1**3`     | `(1, 1, 1)`  | `"e_1**3"`       |
| `e_1*e_2*e_3`| `(1, 2, 3)`  | `"e_1*e_2*e_3"`  |

This representation:
- ✓ Is canonical (always sorted)
- ✓ Naturally handles powers
- ✓ Enables efficient comparison
- ✓ Works seamlessly with dictionaries

## Example Output

```
Number of columns (expansions): 5
  First 2 columns: UW_basis(2)[i] * g(2,2)
  Next 3 columns: UW_basis(3)[i] * g(2,3)

Number of rows (unique monomials): 17

Coefficient matrix C (rows=monomials, columns=expansions):

  e_1**4*e_4                |    0    0   -1    0    0
  e_1**3*e_2*e_3            |    0    0    1    0    0
  e_1**3*e_3                |   -1    0    0    0    0
  e_1**3*e_5                |    0    0    1    0    0
  e_1**2*e_2**2             |    1    0    0    0    0
  e_1**2*e_2*e_4            |    0    0    0   -1    0
  e_1**2*e_4                |    1    0    0    0    0
  e_1*e_2**2*e_3            |    0    0    0    1    0
  e_1*e_2*e_3               |    0   -1    0    0    0
  e_1*e_2*e_5               |    0    0    0    1    0
  e_1*e_3*e_4               |    0    0    0    1   -1
  e_2**3                    |    0    1    0    0    0
  e_2*e_3**2                |    0    0    0    0    1
  e_2*e_4                   |    0    1    0    0    0
  e_3**2                    |    0    1    0    0    0
  e_4**2                    |    0    0    0    0    2
  e_8                       |    0    0    0    0    2
```

## Files Created

1. **e_algebra.py** - Main implementation with full type hints
2. **test_intersection.py** - Complete example showing null space computation
3. **INTERSECTION_README.md** - Comprehensive documentation
4. **IMPLEMENTATION_SUMMARY.md** - This file

## Type Hints

All functions have complete type hints:
- Function signatures use proper typing
- Local variables are annotated for clarity
- Return types are specified
- Full compatibility with modern Python type checkers

## Testing

Run the test suite:
```bash
python src/span_intersection/e_algebra.py          # Full test suite
python src/span_intersection/test_intersection.py  # Intersection example
```

## Key Features

✓ **Correct coefficient extraction** - Handles all term types (E, Mul, Pow, constants)
✓ **Sorted monomial ordering** - Consistent, canonical representation
✓ **Power notation support** - Properly handles `e_i**n` terms
✓ **Integer coefficients** - All operations preserve integer arithmetic
✓ **Full type hints** - Complete type annotations throughout
✓ **Comprehensive documentation** - Detailed docstrings and examples
✓ **Verified correctness** - Matches original implementation results
