#!/usr/bin/env python3
"""
Test the intersect_uw_bases function to find span intersections.
"""

import numpy as np
from e_algebra import e, g, UW_basis, intersect_uw_bases, expand_e

print("=" * 70)
print("Computing Intersection of UW Basis Spaces")
print("=" * 70)
print()

# Define the two bases and their g elements
uw_2 = UW_basis(2)
g_22 = g(2, 2)

uw_3 = UW_basis(3)
g_23 = g(2, 3)

print("Setup:")
print(f"  UW_basis(2) = {uw_2}")
print(f"  g(2,2) = {g_22}")
print()
print(f"  UW_basis(3) = {uw_3}")
print(f"  g(2,3) = {g_23}")
print()

# Show the expansions
print("Expansions:")
print()
print("UW_basis(2)[i] * g(2,2):")
for i, basis_elem in enumerate(uw_2):
    expansion = expand_e(basis_elem * g_22)
    print(f"  [{i}]: {basis_elem} * g(2,2) = {expansion}")
print()

print("UW_basis(3)[j] * g(2,3):")
for j, basis_elem in enumerate(uw_3):
    expansion = expand_e(basis_elem * g_23)
    print(f"  [{j}]: {basis_elem} * g(2,3) = {expansion}")
print()

# Get the coefficient matrix
coeff_matrix, monomials = intersect_uw_bases(uw_2, g_22, uw_3, g_23)

print("=" * 70)
print("Coefficient Matrix")
print("=" * 70)
print()
print(f"Dimensions: {len(coeff_matrix)} rows × {len(coeff_matrix[0])} columns")
print(f"  Columns 0-{len(uw_2)-1}: coefficients from UW_basis(2)[i] * g(2,2)")
print(f"  Columns {len(uw_2)}-{len(uw_2)+len(uw_3)-1}: coefficients from UW_basis(3)[j] * g(2,3)")
print()

# Convert to numpy array for easier manipulation
C = np.array(coeff_matrix, dtype=float)

print("Matrix C:")
for i, row in enumerate(C):
    mon_str = "*".join(f"e_{idx}" for idx in monomials[i]) if monomials[i] else "1"
    row_str = " ".join(f"{int(c):4d}" for c in row)
    print(f"  {mon_str:25s} [{row_str}]")
print()

# Find the null space (solutions to C*x = 0)
print("=" * 70)
print("Finding Null Space (Intersection)")
print("=" * 70)
print()

# Use SVD to find null space
U, s, Vt = np.linalg.svd(C)
rank = np.sum(s > 1e-10)
null_space = Vt[rank:].T

print(f"Rank of matrix: {rank}")
print(f"Dimension of null space: {null_space.shape[1]}")
print()

if null_space.shape[1] > 0:
    print("Null space basis vectors:")
    for i in range(null_space.shape[1]):
        vec = null_space[:, i]
        # Round to clean up numerical errors
        vec = np.round(vec, 10)
        print(f"\n  Vector {i}:")
        print(f"    First {len(uw_2)} components (UW_basis(2) coefficients): {vec[:len(uw_2)]}")
        print(f"    Last {len(uw_3)} components (UW_basis(3) coefficients): {vec[len(uw_2):]}")

        # Verify this is actually a solution
        result = C @ vec
        error = np.linalg.norm(result)
        print(f"    Verification ||C*x||: {error:.2e}")

        # Construct the actual intersection element
        print(f"\n    Intersection element:")
        intersection = None
        for j, coeff in enumerate(vec[:len(uw_2)]):
            if abs(coeff) > 1e-10:
                term = uw_2[j] * g_22 * coeff
                if intersection is None:
                    intersection = term
                else:
                    intersection = intersection + term

        for j, coeff in enumerate(vec[len(uw_2):]):
            if abs(coeff) > 1e-10:
                term = uw_3[j] * g_23 * coeff
                if intersection is None:
                    intersection = term
                else:
                    intersection = intersection + term

        if intersection is not None:
            print(f"      {expand_e(intersection)}")
else:
    print("Null space is empty - intersection is trivial!")

print()
print("=" * 70)
