# Final Implementation: intersect_uw_bases

## Summary

The `intersect_uw_bases` function has been updated to **directly return intersection basis elements** instead of returning the coefficient matrix and monomial keys.

## Function Signature

```python
def intersect_uw_bases(
    UW1_basis: List[Expr],
    g_1: Expr,
    UW2_basis: List[Expr],
    g_2: Expr
) -> List[Expr]
```

## What It Does

1. **Builds coefficient matrix** C where:
   - Each row represents a unique monomial
   - Each column represents an expansion (UW_basis[i] * g)
   - C[j, i] = coefficient of monomial j in expansion i

2. **Solves the homogeneous system** C × x = 0 using SVD to find the null space

3. **Constructs intersection basis elements** from each null space vector:
   - For null vector x = [a₀, ..., a_{m-1}, b₀, ..., b_{n-1}]
   - Returns: sum(aᵢ × UW1_basis[i] × g_1)

4. **Rationalizes coefficients** to clean integer values using:
   - Fraction approximation with limit_denominator
   - LCM of denominators to scale to integers
   - GCD reduction for minimal coefficients

## Example Usage

```python
from e_algebra import UW_basis, g, intersect_uw_bases

# Define the bases and g elements
uw_2 = UW_basis(2)  # [e_1**2, e_2]
g_22 = g(2, 2)      # -e_1*e_3 + e_2**2 + e_4

uw_3 = UW_basis(3)  # [e_1**3, e_1*e_2, e_3]
g_23 = g(2, 3)      # -e_1*e_4 + e_2*e_3 + e_5

# Compute intersection
intersection = intersect_uw_bases(uw_2, g_22, uw_3, g_23)

# Result is a list of SymPy expressions
print(f"Dimension of intersection: {len(intersection)}")
for i, elem in enumerate(intersection):
    print(f"  v_{i} = {elem}")
```

## Example Results

### Test Case 1: UW_basis(2) * g(2,2) ∩ UW_basis(3) * g(2,3)
```
Dimension of intersection: 0
Intersection is trivial (zero only).
```

### Test Case 2: UW_basis(2) * g(2,2) ∩ UW_basis(2) * g(2,2)
```
Dimension of intersection: 2
Intersection basis:
  v_0 = e_1*e_2*e_3 - e_2*e_4 - e_2**3 - e_3**2
  v_1 = e_1**2*e_2**2 + e_1**2*e_4 - e_1**3*e_3
```

As expected, intersecting a space with itself gives the entire space.

## Key Features

✓ **Direct basis computation** - No need to manually process the coefficient matrix
✓ **Integer coefficients** - Rational approximation and GCD reduction for clean results
✓ **Automatic expansion** - All results are fully expanded using the algebra rules
✓ **Proper null space handling** - Uses SVD with numerical tolerance
✓ **Type-safe** - Complete type hints throughout

## Implementation Details

### Null Space Computation
```python
# SVD decomposition
U, s, Vt = np.linalg.svd(C, full_matrices=True)
rank = np.sum(s > 1e-10)
null_space = Vt[rank:].T
```

### Rationalization Process
```python
def _rationalize_vector(vec, max_denom=1000):
    # 1. Convert floats to fractions
    fracs = [Fraction(val).limit_denominator(max_denom) for val in vec]

    # 2. Find LCM of denominators
    lcm = compute_lcm(denominators)

    # 3. Scale to integers
    int_vec = [int(frac * lcm) for frac in fracs]

    # 4. Reduce by GCD
    gcd = compute_gcd(int_vec)
    return [x // gcd for x in int_vec]
```

## Return Value

**Type**: `List[Expr]`

**Contents**:
- Empty list `[]` if intersection is trivial (only contains zero)
- List of SymPy expressions representing a basis for the intersection
- Each basis element is fully expanded and normalized

## Comparison with Original Design

### Before (Old Design)
```python
coeff_matrix, monomials = intersect_uw_bases(uw1, g1, uw2, g2)
# User needs to:
# 1. Convert to numpy
# 2. Compute null space
# 3. Construct basis elements from null vectors
```

### After (Current Design)
```python
intersection = intersect_uw_bases(uw1, g1, uw2, g2)
# Ready to use! Just a list of basis elements
```

## Files

- **e_algebra.py** - Main implementation
- **test_intersection_simple.py** - Simple test examples
- **FINAL_IMPLEMENTATION.md** - This document

## Dependencies

- **SymPy** - For symbolic algebra
- **NumPy** - For null space computation via SVD
- **Python 3.10+** - For modern type hints (| operator)
