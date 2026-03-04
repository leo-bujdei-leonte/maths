import numpy as py
from sympy import symbols, Expr, Symbol, simplify, nsimplify, sympify


print("This program prints the intersection of (B_+)y_n with (B_+)y_m given the total weight(filtration)!")

def check_input(n,m):
    if n >= m:
        raise ValueError(f"n={n} must be less than m={m}!")
    if n < 4:
        raise ValueError(f"n={n} must be at least 4!")

def check_filtr(w):
    if w < 9:
        raise ValueError(f"The filtration must be greater or equal to 9!")


def valid_pairs(x):
    pairs=[]
    for i in range ((x-4)//2 +1):
        for j in range (x-4-2*i +1):
            pairs.append((i,j))
    return pairs




n=int(input("Enter your first weight: "))
m=int(input("Enter your second weight: "))
w=int(input("Enter the desired filtration:"))
check_input(n,m)
check_filtr(w)
a, b = symbols("a b")
terms = []
pairs = valid_pairs(w-m)
for i, j in pairs:
    term = f"{{(a**{i}* (a+b)**{j}) b (1-b)y_{w}}}"
    terms.append(term)
print(f"The intersection is span_{{K}}" ,", ".join(terms))


