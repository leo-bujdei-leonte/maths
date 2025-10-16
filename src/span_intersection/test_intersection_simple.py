#!/usr/bin/env python3
"""
Simple test of the updated intersect_uw_bases function.
"""

from e_algebra import e, g, UW_basis, intersect_uw_bases, expand_e

print("=" * 70)
print("Testing intersect_uw_bases (returns intersection basis directly)")
print("=" * 70)
print()

# Test case 1: UW_basis(2) * g(2,2) ∩ UW_basis(3) * g(2,3)
print("Test 1: UW_basis(2) * g(2,2) ∩ UW_basis(3) * g(2,3)")
print("-" * 70)

uw_2 = UW_basis(2)
g_22 = g(2, 2)
uw_3 = UW_basis(3)
g_23 = g(2, 3)

print(f"UW_basis(2) = {uw_2}")
print(f"g(2,2) = {g_22}")
print()
print(f"UW_basis(3) = {uw_3}")
print(f"g(2,3) = {g_23}")
print()

intersection1 = intersect_uw_bases(uw_2, g_22, uw_3, g_23)

print(f"Result: Dimension of intersection = {len(intersection1)}")
if intersection1:
    print("Intersection basis:")
    for i, elem in enumerate(intersection1):
        print(f"  v_{i} = {elem}")
else:
    print("Intersection is trivial (zero only).")
print()

# Test case 2: UW_basis(3) * g(2,2) ∩ UW_basis(2) * g(3,2)
print("=" * 70)
print("Test 2: UW_basis(3) * g(2,2) ∩ UW_basis(2) * g(3,2)")
print("-" * 70)

g_32 = g(3, 2)

print(f"UW_basis(3) = {uw_3}")
print(f"g(2,2) = {g_22}")
print()
print(f"UW_basis(2) = {uw_2}")
print(f"g(3,2) = {g_32}")
print()

intersection2 = intersect_uw_bases(uw_3, g_22, uw_2, g_32)

print(f"Result: Dimension of intersection = {len(intersection2)}")
if intersection2:
    print("Intersection basis:")
    for i, elem in enumerate(intersection2):
        print(f"  v_{i} = {elem}")
else:
    print("Intersection is trivial (zero only).")
print()

# Test case 3: Simpler example with same g
print("=" * 70)
print("Test 3: UW_basis(2) * g(2,2) ∩ UW_basis(2) * g(2,2)")
print("(Should give the entire space)")
print("-" * 70)

intersection3 = intersect_uw_bases(uw_2, g_22, uw_2, g_22)

print(f"Result: Dimension of intersection = {len(intersection3)}")
if intersection3:
    print("Intersection basis:")
    for i, elem in enumerate(intersection3):
        print(f"  v_{i} = {elem}")
else:
    print("Intersection is trivial (zero only).")
print()

print("=" * 70)
print("Summary:")
print("  The intersect_uw_bases function now directly returns")
print("  a list of SymPy expressions representing the intersection basis.")
print("=" * 70)
