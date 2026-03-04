#!/usr/bin/env python3
"""
Demo of e_algebra.py: Asymmetric multiplication algebra for e_i elements

Rule: e_n * e_m = e_m * e_n + (m-n) * e_(n+m) when m > n
"""

from e_algebra import e, expand_e, g, UW_basis

print("=" * 70)
print("e_algebra.py - Asymmetric Multiplication Algebra Demo")
print("=" * 70)
print()

# Basic operations
print("1. Basic multiplication (automatic normalization):")
print("-" * 70)
print(f"   e(2) * e(1) = {expand_e(e(2) * e(1))}")
print(f"   e(3) * e(1) = {expand_e(e(3) * e(1))}")
print(f"   e(3) * e(2) = {expand_e(e(3) * e(2))}")
print()

# Complex products
print("2. Complex products:")
print("-" * 70)
print(f"   e(3) * e(2) * e(1) = {expand_e(e(3) * e(2) * e(1))}")
print()

# Powers (same index - no simplification)
print("3. Powers of same element:")
print("-" * 70)
print(f"   e(2) * e(2) = {expand_e(e(2) * e(2))}")
print(f"   e(1)**3 = {expand_e(e(1)**3)}")
print("   (Powers stay as-is since rule only applies when indices differ)")
print()

# Addition and scalar multiplication
print("4. Addition and scalar multiplication:")
print("-" * 70)
print(f"   e(1) + 2*e(2) + 3*e(1) = {expand_e(e(1) + 2*e(2) + 3*e(1))}")
print(f"   (e(1) + e(2)) * e(3) = {expand_e((e(1) + e(2)) * e(3))}")
print()

# g(n, m) function
print("5. The g(n, m) function:")
print("-" * 70)
print("   g(n, m) = e(n) * e(m) - e(1) * e(n+m-1) + e(n+m) * (n-1)")
print()
print(f"   g(2, 2) = {g(2, 2)}")
print(f"   g(2, 3) = {g(2, 3)}")
print()

# UW_basis
print("6. UW basis generation:")
print("-" * 70)
print("   UW_basis(n) returns products of e_i where indices sum to n")
print()
print("   UW_basis(2):")
for i, elem in enumerate(UW_basis(2)):
    print(f"      [{i}]: {elem}")
print()
print("   UW_basis(3):")
for i, elem in enumerate(UW_basis(3)):
    print(f"      [{i}]: {elem}")
print()
print("   UW_basis(4):")
for i, elem in enumerate(UW_basis(4)):
    print(f"      [{i}]: {elem}")
print()

# Interaction between g and basis
print("7. Products involving g(n, m):")
print("-" * 70)
g_22 = g(2, 2)
print(f"   e(3) * g(2, 2) = {expand_e(e(3) * g_22)}")
print()

print("=" * 70)
print("All expressions are automatically normalized to increasing index order!")
print("=" * 70)
