import sys
import sympy
from __future__ import annotations
import typer
from pathlib import Path
from sympy import symbols, Expr, Symbol, simplify, nsimplify, sympify
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from copy import deepcopy
from collections import defaultdict

"""
    Computes the product e_a1 * e_a2 * ... * e_an, where e_ai = x^{ai+1} partial.
    The inputs are a list of coefficients a1, a2, ..., an.
    """
def main(coefficients):
 # define symbolic variable for computation
    X = sympy.symbols('X')

    # define partial 'polynomial' as a list of coefficients
    # expression[i] is the coefficient of partial^i
    expression = [0, - X**(coefficients[-1]+1)]

def get_basis_partition(n:int) -> list[tuple[int]]:
    def partitions(n, I=1):
        yield (n,)
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, i):
                yield (i,) + p

    # return every partition of n into a sum of integers in increasing order
    # e.g. 3 -> [[1, 1, 1], [1, 2], [3]]
    return list(partitions(n))

  # iteratively multiply the partials right-to-left
for coefficient in coefficients[:-1][::-1]:
    new_term = - X**(coefficient+1)

        # add the new partials
    for i in range(0, len(expression)-1):
        expression[i] = expression[i] + sympy.diff(expression[i+1])
        
        # multiply with the new coefficient
    expression = [new_term * term for term in expression]
        
        # all coefficients are now shifted left; insert a new 0
    expression.insert(0, 0)

    # turn the resulting expression into a string expression
s = str(expression[1]) + " d"
for i in range(2, len(expression)):
    s += " + " + str(expression[i]) + " d^" + str(i)
print(s)


def compute_generic_witt_map(n: int, basis_partition: list[tuple[int]]) -> expression:
    """
    Constructs an expression as a linear combination of partitions of n, with symbolic coefficents alpha_i.
    """
    a, b = symbols('a b')
    alphas = symbols(' '.join([f'alpha_{i}' for i in range(len(basis_partition))]))
    if n == 1:
        alphas = [alphas]
    generic_witt_map = expression(sympify('0'), n)
    for alpha, partition in zip(alphas, basis_partition):
        mapp = expression(sympify('1'), 0)
        for i in range(len(partition)):
            expr= expression(-a-partition[i]*b,partition[i])
            mapp *= expr 
        generic_witt_map = generic_witt_map + mapp * alpha
    
    return generic_witt_map



    

if __name__ == "__main__":
    coefficients = [int(coef) for coef in sys.argv[1:]]
    main(coefficients)




def main(coefficients):
    """
    Computes the product e_a1 * e_a2 * ... * e_an, where e_ai = x^{ai+1} partial.
    The inputs are a list of coefficients a1, a2, ..., an.
    """



    # define partial 'polynomial' as a list of coefficients
    # expression[i] is the coefficient of partial^i
    expression = [0, - X**(coefficients[-1]+1)]

    # iteratively multiply the partials right-to-left
    for coefficient in coefficients[:-1][::-1]:
        new_term = - X**(coefficient+1)

        # add the new partials
        for i in range(0, len(expression)-1):
            expression[i] = expression[i] + sympy.diff(expression[i+1])
        
        # multiply with the new coefficient
        expression = [new_term * term for term in expression]
        
        # all coefficients are now shifted left; insert a new 0
        expression.insert(0, 0)

    # turn the resulting expression into a string expression
    s = str(expression[1]) + " d"
    for i in range(2, len(expression)):
        s += " + " + str(expression[i]) + " d^" + str(i)
    print(s)


if __name__ == "__main__":
    coefficients = [int(coef) for coef in sys.argv[1:]]
    main(coefficients)


def check_input(n,m):
    if n < 1 :
        raise ValueError(f"n={n} must be at least 1!")
    if m < 1:
        raise ValueError(f"m={n} must be at least 1!")

def check_filtr(w):
    if w < 3:
        raise ValueError(f"The filtration must be greater or equal to 3!")



n=int(input("What is n? "))
m=int(input("What is m? "))
w=int(input("Enter the desired filtration:"))
check_input(n,m)
check_filtr(w)

terms = []

print(f"The intersection is {{K}}" ,", ".)

