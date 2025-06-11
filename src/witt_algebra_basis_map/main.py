"""
TODO description and docstrings
"""
from __future__ import annotations
import typer
from pathlib import Path
from sympy import symbols, Expr, Symbol
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from copy import deepcopy

app = typer.Typer(pretty_exceptions_enable=False)


class Expression:
    a_b_expr: Expr
    n: int

    @classmethod
    def from_str(cls, s: str) -> Expression:
        # TODO for multiple t, split by + after splitting by t
        a_b_substr, n_substr = s.split('t')

        a_b_substr = "*".join(a_b_substr.split('*')[:-1])
        a_b_expr = parse_expr(a_b_substr) # , evaluate=False)

        n = int(n_substr[2:])

        return cls(a_b_expr, n)

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


def get_basis_partition(n) -> list[tuple[int]]:
    def partitions(n, I=1):
        yield (n,)
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, i):
                yield (i,) + p

    # return every partition of n into a sum of integers in increasing order
    # e.g. 3 -> [[1, 1, 1], [1, 2], [3]]
    return list(partitions(n))

def compute_generic_witt_map(n: int, basis_partition: list[tuple[int]]) -> Expression:
    a, b = symbols('a b')
    alphas = symbols(' '.join([f'alpha_{i}' for i in range(len(basis_partition))]))
    generic_witt_map = Expression(symbols('0'), n)

    for alpha, partition in zip(alphas, basis_partition):
        mapp = Expression(symbols('1'), 0)
        for i in range(len(partition)):
            expr= Expression(a+partition[i]*b,partition[i])
            mapp *= expr 
        generic_witt_map = generic_witt_map + mapp * alpha
    
    return generic_witt_map

def get_alpha_coefficients_matrix(n: int, expr: Expression, basis_partition: list[Symbol]) -> np.ndarray:
    res = [[None] * (n+1) for _ in range(n+1)]
    a, b = symbols('a b')

    for i in range(n+1):
        for j in range(n+1):
            res[i][j] = basis_partition.coeff()
        print(res[i][j])

def get_expanded_coefficients_vector(expr: Expression, basis_partition: list[Symbol]) -> np.ndarray:
    ...

def solve_linear_system(M: np.ndarray, b: np.ndarray) -> np.ndarray:
    ...

@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to the input file containing the expression."),
):
    # TODO handle multiple t
    expr = Expression.from_str(input_file.open().read())
    print(f"Found expression: {expr}")
    basis_partition = get_basis_partition(expr.n)
    expr_with_alphas = compute_generic_witt_map(expr.n, basis_partition)
    print(expr_with_alphas)
    
    # M = get_alpha_coefficients_matrix(expr.n, expr_with_alphas, basis_partition)
    # b = get_expanded_coefficients_vector(expr, basis_partition)

    # alphas = solve_linear_system(M, b)
    # result = Expression.from_coefficients(alphas, basis_partition)

    # print(result)

if __name__ == "__main__":
    app()