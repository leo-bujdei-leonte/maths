'''
import sys
import sympy 
import math

# Define symbolic variables
x = sympy.symbols('x')
n = sympy.symbols('n', integer=True)
d = sympy.symbols('∂', commutative=False) # Differential operator
v = sympy.IndexedBase('v')
# --- Core Simplification Rule ---
# Implements the commutator: ∂ * x^n = x^n * ∂ + n * x^(n-1)

def expand_derivation(expr):
    expr = sympy.expand(expr)
    
    changed = True
    while changed:
        changed = False
        
        if expr.is_Mul:
            # Get the factors - call it "factors" consistently
            factors = list(expr.args)
            
            # Look for d followed by x^n
            for i in range(len(factors) - 1):
                if factors[i] == d:
                    next_factor = factors[i+1]
                    
                    if next_factor == x:
                        # d*x = x*d + 1
                        left = sympy.Mul(*factors[:i]) if i > 0 else 1
                        right = sympy.Mul(*factors[i+2:]) if i+2 < len(factors) else 1
                        expr = sympy.expand(left * (x*d + 1) * right)
                        changed = True
                        break
                    
                    elif next_factor.is_Pow and next_factor.base == x:
                        # d*x^n = x^n*d + n*x^(n-1)
                        n = next_factor.exp
                        left = sympy.Mul(*factors[:i]) if i > 0 else 1
                        right = sympy.Mul(*factors[i+2:]) if i+2 < len(factors) else 1
                        expr = sympy.expand(left * (x**n * d + n * x**(n-1)) * right)
                        changed = True
                        break
                        
        elif expr.is_Add:
            # Recursively process each term
            new_terms = [expand_derivation(term) for term in expr.args]
            new_expr = sympy.Add(*new_terms)
            if new_expr != expr:
                expr = new_expr
                changed = True
                
    return expr

# Operator Definitions 

# Operator e_n: e(n) = x^(n+1) * ∂
def e(n_value):
    return x**(n_value + 1) * d

def f(n_value):
    return x**(n_value + 1)

# Operator function for Orbit homomorphisms
def orbit_homomorphism(a,n):
    term1=e(a)
    term2 = 0
    for i in range(0,n):
        term2= term2 + sympy.diff(f(a),x,i+1)*v[i]/math.factorial(i+1)
    return expand_derivation(term1 + term2)
   

# Operator Multiplication: Apply simplification after multiplication
def multiply_operators(a, b):
    return expand_derivation(a * b)

# Operator g_n,m: g(n,m) = e(n)e(m) - e(n-1)e(m+1) + e(n+m)
def g(n_val, m_val):
    term1 = multiply_operators(e(n_val), e(m_val))
    term2 = multiply_operators(e(n_val - 1), e(m_val + 1))
    term3 = e(n_val + m_val)
    return expand_derivation(term1 - term2 + term3)

def homomorphism_calculated(n,m,i):
    return expand_derivation(orbit_homomorphism(2,i)*orbit_homomorphism(2,i))
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

# Try the orbit homomorphism now
orbit_result = orbit_homomorphism(2,1)
print(f"The result for the orbit homomorphism e(2) with n=1 is:")
print(sympy.expand(orbit_result))
print(f"The result for the orbit homomorphism is")
print(expand_derivation(homomorphism_calculated(2,2,1)))
'''

import sympy 
import math

# Define symbolic variables
x = sympy.symbols('x')
d = sympy.symbols('∂', commutative=False)
v = sympy.IndexedBase('v')

def flatten_mul(expr):
    """Flatten nested Mul expressions"""
    if not expr.is_Mul:
        return [expr]
    
    result = []
    for arg in expr.args:
        if arg.is_Mul:
            result.extend(flatten_mul(arg))
        else:
            result.append(arg)
    return result

def apply_one_commutation(expr):
    """Apply a single [∂,x]=1 commutation rule"""
    if not expr.is_Mul:
        return expr
    
    factors = flatten_mul(expr)
    
    for i in range(len(factors) - 1):
        if factors[i] == d:
            j = i + 1
            coeff_factors = []
            x_power = None
            
            while j < len(factors):
                f = factors[j]
                if f == x:
                    x_power = 1
                    j += 1
                    break
                elif f.is_Pow and f.base == x:
                    x_power = f.exp
                    j += 1
                    break
                elif f.is_number or (f.is_Mul and not any(arg.has(x) for arg in f.args)):
                    coeff_factors.append(f)
                    j += 1
                else:
                    break
            
            if x_power is not None:
                coeff = sympy.Mul(*coeff_factors) if coeff_factors else 1
                left = sympy.Mul(*factors[:i]) if i > 0 else 1
                right = sympy.Mul(*factors[j:]) if j < len(factors) else 1
                
                if x_power == 1:
                    return left * coeff * x * d * right + left * coeff * right
                else:
                    return (left * coeff * x**x_power * d * right + 
                           left * coeff * x_power * x**(x_power-1) * right)
    
    return expr

def cleanup_mul_terms(expr):
    """Simplify products by evaluating commutative parts"""
    if not expr.is_Mul:
        return expr
    
    factors = flatten_mul(expr)
    # Group commutative and non-commutative parts
    commutative = []
    non_commutative = []
    
    for f in factors:
        if f == d or (f.is_Mul and d in flatten_mul(f)):
            non_commutative.append(f)
        else:
            commutative.append(f)
    
    # Multiply commutative parts
    if commutative:
        comm_product = sympy.Mul(*commutative, evaluate=True)
    else:
        comm_product = 1
    
    if non_commutative:
        non_comm_product = sympy.Mul(*non_commutative, evaluate=False)
        return comm_product * non_comm_product
    else:
        return comm_product

def weyl_simplify(expr):
    """Simplify using Weyl algebra [∂,x]=1"""
    changed = True
    iterations = 0
    
    while changed and iterations < 150:
        changed = False
        iterations += 1
        
        if expr.is_Add:
            new_terms = []
            for term in expr.args:
                new_term = apply_one_commutation(term)
                new_term = cleanup_mul_terms(new_term)
                if new_term != term:
                    changed = True
                new_terms.append(new_term)
            expr = sympy.Add(*new_terms)
        else:
            new_expr = apply_one_commutation(expr)
            new_expr = cleanup_mul_terms(new_expr)
            if new_expr != expr:
                changed = True
                expr = new_expr
    
    return expr

def weyl_multiply(expr1, expr2):
    """
    Multiply two expressions in Weyl algebra
    """
    if expr1.is_Add or expr2.is_Add:
        if expr1.is_Add:
            return sympy.Add(*[weyl_multiply(t, expr2) for t in expr1.args])
        else:
            return sympy.Add(*[weyl_multiply(expr1, t) for t in expr2.args])
    
    factors1 = flatten_mul(expr1) if expr1.is_Mul else [expr1]
    factors2 = flatten_mul(expr2) if expr2.is_Mul else [expr2]
    
    product = sympy.Mul(*(factors1 + factors2), evaluate=False)
    result = weyl_simplify(product)
    
    return result

def e(n_value):
    return x**(n_value + 1) * d

def f(n_value):
    return x**(n_value + 1)

def orbit_homomorphism(a, n):
    term1 = e(a)
    term2 = 0
    for i in range(0, n):
        term2 = term2 + sympy.diff(f(a), x, i+1) * v[i] / math.factorial(i+1)
    result = term1 + term2
    return weyl_simplify(result)

def multiply_operators(a, b):
    return weyl_multiply(a, b)

def g(n_val, m_val):
    term1 = multiply_operators(e(n_val), e(m_val))
    term2 = multiply_operators(e(n_val - 1), e(m_val + 1))
    term3 = e(n_val + m_val)
    return weyl_simplify(term1 - term2 + term3)

def homomorphism_calculated(n, m, i):
    psi1 = orbit_homomorphism(n, i)
    psi2 = orbit_homomorphism(m,i)
    result = multiply_operators(psi1, psi2)
    return weyl_simplify(result)

# --- Testing ---
print("Orbit Homomorphism Results with Weyl Algebra [∂,x]=1:")
print("=" * 60)

psi_2_1 = orbit_homomorphism(2, 1)
print(f"ψ(2,1) = {psi_2_1}")
print()

result1 = homomorphism_calculated(2, 2, 1)
print(f"ψ(2,1)^2 =")
print(result1)

result2 = homomorphism_calculated(1, 3, 1)
print(f"ψ(2,1)^2 =")
print(result2)
total = sympy.expand(result1-result2+orbit_homomorphism(4,1))
print(total)