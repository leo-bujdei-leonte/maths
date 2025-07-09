from sympy import symbols, simplify, expand

# Define symbols
a, b, t = symbols('a b t')

# Define the sets
x = {
    b**2*(1-b)**2*t**9,
    (a+b)*b**2*(1-b)**2*t**9,
    (a+b)*(a+b+1)*(a+b+2)*(a+b+3)*(a+b+4)*b*(1-b)*t**9,
    (a+b)*(a+b+1)*(a+3*b+2)*b*(1-b)*t**9,
    (a+b)*(a+b+1)*(a+b+2)*(a+2*b+3)*b*(1-b)*t**9,
    (a+b)*(a+4*b+1)*b*(1-b)*t**9,
    (a+5*b)*b*(1-b)*t**9
}

y = {
    b**2*(1-b)**2*t**9,
    (a+b)*(a+b+1)*(a+b+2)*(a+b+3)*b*(1-b)*t**9,
    (a+b)*(a+3*b+1)*b*(1-b)*t**9,
    (a+b)*(a+b+1)*(a+2*b+2)*b*(1-b)*t**9,
    (a+4*b)*b*(1-b)*t**9
}

# Use simplified expressions for accurate comparison
x_simplified = set(simplify(expand(expr)) for expr in x)
y_simplified = set(simplify(expand(expr)) for expr in y)

# Compute the intersection
z = x_simplified.intersection(y_simplified)

# Print the simplified y set
#for expr in sorted(y_simplified, key=str):
#    print(expr)


print("\nIntersection of the sets:")
for expr in sorted(z, key=str):
    print(expr)
