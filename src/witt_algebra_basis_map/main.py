"""
TODO description and docstrings
"""
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

    def __mul__(self, other: Expression) -> Expression:
        res = deepcopy(self)
        res.a_b_expr = self.a_b_expr * other.a_b_expr.subs('a', f'a + {self.n}')
        res.n += other.n

        return res


def get_basis_partition(n: int) -> list[Symbol]:
    ...

def compute_generic_witt_map(basis_partition: list[Symbol]) -> Expression:
    ...

def get_alpha_coefficients_matrix(expr: Expression, basis_partition: list[Symbol]) -> np.ndarray:
    ...

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
    basis_partition = get_basis_partition(expr.n)
    expr_with_alphas = compute_generic_witt_map(basis_partition)

    M = get_alpha_coefficients_matrix(expr_with_alphas, basis_partition)
    b = get_expanded_coefficients_vector(expr, basis_partition)

    alphas = solve_linear_system(M, b)
    result = Expression.from_coefficients(alphas, basis_partition)

    print(result)

if __name__ == "__main__":
    app()