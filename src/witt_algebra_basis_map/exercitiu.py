from __future__ import annotations
import typer
from pathlib import Path
from sympy import symbols, Expr, Symbol
import numpy as np
from copy import deepcopy

app = typer.Typer(pretty_exceptions_enable=False)


class Expression:
    a_b_expr: Expr
    n: int

    def __init__(self, a_b_expr: Expr, n: int):
        self.a_b_expr = a_b_expr
        self.n = n
    
    def __str__(self) -> str:
        return "(" + str(self.a_b_expr.expand()) + f") * t^{self.n}"

    def __mul__(self, other: Expression | Symbol) -> Expression:
        res = deepcopy(self)

        if isinstance(other, Expression):
            res.a_b_expr = self.a_b_expr * other.a_b_expr.subs('a', f'a + {self.n}')
            res.n += other.n
        
        elif isinstance(other, Symbol):
            res.a_b_expr = self.a_b_expr * other

        return res
    
    def __add__(self, other: Expression) -> Expression:
        res = deepcopy(self)

        if self.n == other.n:
            res.a_b_expr += other.a_b_expr
        else:
            raise NotImplementedError()
        
        return res




def get_basis_partition(n) -> list[list[int]]:
    # return every partition of n into a sum of integers in increasing order
    # e.g. 3 -> [[1, 1, 1], [1, 2], [3]]
    return [[1, 1, 1], [1, 2], [3]]

def compute_generic_witt_map(basis_partition):
    a, b = symbols('a b')
    alphas = symbols(' '.join([f'alpha_{i}' for i in range(len(basis_partition))]))
    generic_witt_map = Expression(symbols('0'), 3)

    for alpha, partition in zip(alphas, basis_partition):
        mapp = Expression(symbols('1'), 0)
        for i in range(len(partition)):
            expr= Expression(a+partition[i]*b,partition[i])
            mapp *= expr 
        generic_witt_map = generic_witt_map + mapp * alpha
    
    return generic_witt_map
    
basis_partition = get_basis_partition(3)
expr_with_alphas = compute_generic_witt_map(basis_partition)
print(expr_with_alphas)