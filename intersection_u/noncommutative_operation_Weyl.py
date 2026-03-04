import sys
import sympy

# Define symbolic variables
x = sympy.symbols('x')
n = sympy.symbols('n', integer=True)
d = sympy.symbols('∂', commutative=False) # Differential operator

# --- Core Simplification Rule ---
# Implements the commutator: ∂ * x^n = x^n * ∂ + n * x^(n-1)
def expand_derivation(expr):
    expr = sympy.expand(expr)

    if expr.is_Mul:
        factors = list(expr.args)
        for i in range(len(factors) - 1):
            if factors[i] == d and factors[i+1].is_Pow and factors[i+1].base == x:
                n_val = factors[i+1].exp
                
                # Apply the rule: d * x^n -> (x^n * d + n * x^(n-1))
                new_term = (x**n_val * d + n_val * x**(n_val - 1))
                
                # Reconstruct the expression with the substitution and continue expanding
                return expand_derivation(sympy.Mul(*factors[:i]) * new_term * sympy.Mul(*factors[i+2:]))
    
    elif expr.is_Add:
        return sympy.Add(*[expand_derivation(arg) for arg in expr.args])
    
    return expr


# --- Operator Definitions ---

# Operator e_n: e(n) = x^(n+1) * ∂
def e(n_value):
    return x**(n_value + 1) * d

# Operator Multiplication: Apply simplification after multiplication
def multiply_operators(a, b):
    return expand_derivation(a * b)

# Operator g_n,m: g(n,m) = e(n)e(m) - e(n-1)e(m+1) + e(n+m)
def g(n_val, m_val):
    term1 = multiply_operators(e(n_val), e(m_val))
    term2 = multiply_operators(e(n_val - 1), e(m_val + 1))
    term3 = e(n_val + m_val)
    return expand_derivation(term1 - term2 + term3)

# --- Calculation of e_1^3 * g_2,2 ---

# 1. Calculate g_2,2
g_22 = g(2, 2)

# 2. Calculate e_1^3 = e_1 * e_1 * e_1
e_1 = e(1)
e_1_sq = multiply_operators(e_1, e_1)
e_1_cubed = multiply_operators(e_1_sq, e_1)

# 3. Calculate e_1^3 * g_22
e_1_cubed_g_22 = multiply_operators(e_1_cubed, g_22)

print(f"g_22 (for reference): {g_22}")
print("-" * 20)
print(f"The result for e_1^3 g_22 is:")
print(sympy.expand(e_1_cubed_g_22))


# --- Calculation of e_1^2 * g_2,3 ---

# 1. Calculate g_2,3
g_23 = g(2, 3)

# 2. Calculate e_1^2 = e_1 * e_1
e_1 = e(1)
e_1_sq = multiply_operators(e_1, e_1)

# 3. Calculate e_1^2 * g_23
e_1_squared_g_23 = multiply_operators(e_1_sq, g_23)

print(f"g_23 (for reference): {g_23}")
print("-" * 20)
print(f"The result for e_1^2 g_23 is:")
print(sympy.expand(e_1_squared_g_23))