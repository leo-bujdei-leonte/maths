import numpy as py
from sympy import symbols, Expr, Symbol, simplify, nsimplify, sympify


print("This program prints the intersection of (B_+)y_n with (B_+)y_m!")

def check_input(n,m):
    if n >= m:
        raise ValueError(f"n={n} must be less than m={m}")
    if n < 4:
        raise ValueError(f"n={n} must be at least 4")


def valid_pairs(x):
    pairs=[]
    for i in range ((x-4)//2 +1):
        for j in range (x-4-2*i +1):
            pairs.append((i,j))
    return pairs




n=int(input("Enter your first weight: "))
m=int(input("Enter your second weight: "))
check_input(n,m)
a, b = symbols("a b")
terms = []
pairs = valid_pairs(n)
for i, j in pairs:
    term = f"{{(a**{i}* (a+b)**{j}) b (1-b)y_{n+m}}}"
    terms.append(term)
print(f"The intersection is span_{{K}}" ,", ".join(terms))


