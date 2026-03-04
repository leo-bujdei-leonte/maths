# Computing Span Intersections with e_algebra.py

## Overview

The `intersect_uw_bases` function computes a coefficient matrix that can be used to find the intersection of two subspaces of the form:

- **Subspace 1**: Span of `{UW1_basis[i] * g_1 : i = 0, ..., m-1}`
- **Subspace 2**: Span of `{UW2_basis[j] * g_2 : j = 0, ..., n-1}`

## Mathematical Setup

### The Intersection Problem

An element is in the intersection if it can be expressed in both forms:
```
sum(a_i * UW1_basis[i] * g_1) = sum(b_j * UW2_basis[j] * g_2)
```

This is equivalent to:
```
sum(a_i * UW1_basis[i] * g_1) - sum(b_j * UW2_basis[j] * g_2) = 0
```

### Coefficient Matrix Construction

The function constructs a matrix **C** where:
- **Rows** represent unique monomials (products of e_i elements)
- **Columns** represent the expansions:
  - First `len(UW1_basis)` columns: expansions of `UW1_basis[i] * g_1`
  - Next `len(UW2_basis)` columns: expansions of `UW2_basis[j] * g_2`
- **Entry C[j, i]**: coefficient of the j-th monomial in the i-th expansion

### Finding the Intersection

To find the intersection basis:

1. **Solve the homogeneous system**: `C * x = 0`
2. **Find the null space** of C
3. **Each null vector** `x = [a_0, ..., a_{m-1}, b_0, ..., b_{n-1}]` gives an intersection element:
   ```
   intersection_element = sum(a_i * UW1_basis[i] * g_1)
                        = -sum(b_j * UW2_basis[j] * g_2)
   ```

## Example Usage

```python
from e_algebra import e, g, UW_basis, intersect_uw_bases
import numpy as np

# Define the bases
uw_2 = UW_basis(2)  # [e_1**2, e_2]
g_22 = g(2, 2)      # -e_1*e_3 + e_2**2 + e_4

uw_3 = UW_basis(3)  # [e_1**3, e_1*e_2, e_3]
g_23 = g(2, 3)      # -e_1*e_4 + e_2*e_3 + e_5

# Compute coefficient matrix
coeff_matrix, monomials = intersect_uw_bases(uw_2, g_22, uw_3, g_23)

# Convert to numpy and find null space
C = np.array(coeff_matrix, dtype=float)
U, s, Vt = np.linalg.svd(C)
rank = np.sum(s > 1e-10)
null_space = Vt[rank:].T

# Each column of null_space is a solution vector
print(f"Dimension of intersection: {null_space.shape[1]}")
```

## Matrix Structure Example

For the intersection of `UW_basis(2) * g(2,2)` and `UW_basis(3) * g(2,3)`:

```
Coefficient Matrix C (17 monomials × 5 expansions):

Monomial              | Col 0 | Col 1 | Col 2 | Col 3 | Col 4
                      | uw2[0]| uw2[1]| uw3[0]| uw3[1]| uw3[2]
                      | *g22  | *g22  | *g23  | *g23  | *g23
--------------------- | ----- | ----- | ----- | ----- | -----
e_1^3 e_3             |  -1   |   0   |   0   |   0   |   0
e_1^2 e_2^2           |   1   |   0   |   0   |   0   |   0
e_1^2 e_4             |   1   |   0   |   0   |   0   |   0
e_1 e_2 e_3           |   0   |  -1   |   0   |   0   |   0
e_2^3                 |   0   |   1   |   0   |   0   |   0
e_2 e_4               |   0   |   1   |   0   |   0   |   0
e_3^2                 |   0   |   1   |   0   |   0   |   0
e_1^4 e_4             |   0   |   0   |  -1   |   0   |   0
e_1^3 e_2 e_3         |   0   |   0   |   1   |   0   |   0
e_1^3 e_5             |   0   |   0   |   1   |   0   |   0
e_1^2 e_2 e_4         |   0   |   0   |   0   |  -1   |   0
e_1 e_2^2 e_3         |   0   |   0   |   0   |   1   |   0
e_1 e_2 e_5           |   0   |   0   |   0   |   1   |   0
e_1 e_3 e_4           |   0   |   0   |   0   |   1   |  -1
e_2 e_3^2             |   0   |   0   |   0   |   0   |   1
e_4^2                 |   0   |   0   |   0   |   0   |   2
e_8                   |   0   |   0   |   0   |   0   |   2
```

In this example, the matrix has full rank (rank 5), so the null space is empty and the intersection is trivial.

## Implementation Details

### Function Signature

```python
def intersect_uw_bases(
    UW1_basis: List[Expr],
    g_1: Expr,
    UW2_basis: List[Expr],
    g_2: Expr
) -> Tuple[List[List[int]], List[Tuple[int, ...]]]
```

### Return Values

1. **coefficient_matrix**: A list of lists representing the coefficient matrix C
   - `coefficient_matrix[j][i]` = coefficient of monomial j in expansion i

2. **monomial_keys**: Ordered list of monomial representations
   - Each monomial is a tuple of sorted indices, e.g., `(1, 2, 3)` for `e_1*e_2*e_3`
   - Powers are represented by repeated indices, e.g., `(2, 2)` for `e_2**2`

### Helper Functions

- `_extract_monomials(expr)`: Extracts all monomials and coefficients from an expression
- `_extract_single_term(term)`: Extracts monomial key and coefficient from a single term

## Monomial Representation

Monomials are represented as **sorted tuples of indices**:

| Expression | Monomial Key |
|------------|--------------|
| `e_1`      | `(1,)`       |
| `e_1*e_2`  | `(1, 2)`     |
| `e_2**2`   | `(2, 2)`     |
| `e_1**3`   | `(1, 1, 1)`  |
| `e_1*e_2*e_3` | `(1, 2, 3)` |

This representation:
- Is canonical (always sorted)
- Naturally handles powers
- Enables efficient comparison and hashing

## Notes

- The algebra rule `e_n * e_m = e_m * e_n + (m-n) * e_(n+m)` when `m > n` automatically normalizes all products
- All expressions are expanded before coefficient extraction
- The coefficient matrix has integer entries since all operations preserve integer coefficients
- An empty null space indicates a trivial intersection (only the zero element)
