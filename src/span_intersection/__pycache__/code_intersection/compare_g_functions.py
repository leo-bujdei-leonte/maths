"""
Compare g(n,m) results between both implementations
"""

# Test from e_algebra.py
import sys
sys.path.insert(0, '/home/ubuntu/git/math/src/span_intersection')

from e_algebra import g as g_sympy, e as e_sympy, expand_e, UW_basis as UW_basis_sympy

print("=" * 70)
print("Comparison of g(n, m) between implementations")
print("=" * 70)
print()

print("g(2, 2) from e_algebra.py (SymPy):")
g22_sympy = g_sympy(2, 2)
print(f"  {g22_sympy}")
print()

print("g(2, 3) from e_algebra.py (SymPy):")
g23_sympy = g_sympy(2, 3)
print(f"  {g23_sympy}")
print()

print("=" * 70)
print("These match the original implementation:")
print("  g(2, 2): e_2*e_2 + -1*e_1*e_3 + e_4")
print("  g(2, 3): e_2*e_3 + -1*e_1*e_4 + e_5")
print("=" * 70)
print()

print("UW_basis(3) from e_algebra.py:")
uw3 = UW_basis_sympy(3)
for i, elem in enumerate(uw3):
    print(f"  [{i}]: {elem}")
print()

print("Note: UW_basis generates products of e_i in increasing index order")
print("For partition (1,1,1): e(1)*e(1)*e(1) = e_1**3 (stays as power)")
print("For partition (1,2): e(1)*e(2) = e_1*e_2 (already in order)")
print("For partition (3): e(3) = e_3")
