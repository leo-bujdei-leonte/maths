"""
Let \psi: U(W_+) -> K[a,b][\sigma,t] be a map that takes the basis elements of U(W_+)
and send them to an expression involving a,b,t, that is, \psi(e_n) = -(a+nb)t**n.

This program takes a computed form of \psi(\sum_i e_i) and it computes the original e_i 
in an increasing power bases on the partitions of the power of 't'.
 

Usage: python script.py <input_file>

The input file must contain a string of the form: (expression in a and b )* t**n

Example: (a**2 + 2ab + b**3)* t**2
"""
from __future__ import annotations
import typer
from pathlib import Path
from sympy import symbols, Expr, Symbol, simplify, nsimplify, sympify
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from copy import deepcopy
from collections import defaultdict

app = typer.Typer(pretty_exceptions_enable=False)


class Expression:
    a_b_expr: Expr
    n: int
    """
    Takes a string of the form <expr in a and b>* t**n 
    and outputs an expression object.

    Methods:
    from_str(): Parses a string of the form <a_b_expr>*t**n into an Expression.
    _str_(): Returns a string representation.
    _mul_(): Defines a noncommutative multiplication.
    _add_(): Adds 2 expression with the same power of 't'. 
    """
    @classmethod
    def from_str(cls, s: str) -> Expression:
        """
        Parse the given string into an an expression with 'a' and 'b' and the power of 't'.
        """
        # TODO for multiple t, split by + after splitting by t
        if 't' not in s:
            raise ValueError("Cannot find 't' term in equation")
        a_b_substr, n_substr = s.split('t')

        a_b_substr = "*".join(a_b_substr.split('*')[:-1])
        a_b_expr = parse_expr(a_b_substr) # , evaluate=False)

        if n_substr:
            n = int(n_substr[2:])
        else:
            n = 1

        return cls(a_b_expr, n)

    def __init__(self, a_b_expr: Expr, n: int):
        self.a_b_expr = a_b_expr
        self.n = n
    
    def __str__(self) -> str:
        return "(" + str(self.a_b_expr.expand()) + f") * t^{self.n}"

    def __mul__(self, other: Expression | Symbol) -> Expression:
        res = deepcopy(self)
        """
        Defines the noncommutative multiplication.

        """
        if isinstance(other, Expression):
            res.a_b_expr = self.a_b_expr * other.a_b_expr.subs('a', f'a + {self.n}')
            res.n += other.n
        
        elif isinstance(other, Symbol):
            res.a_b_expr = self.a_b_expr * other

        return res
    
    def __add__(self, other: Expression) -> Expression:
        """
        Adds two expressions with the same power of t.

        Raises: 
        NotImplementedError: if the degrees of t does not match.

        Returns: Sum of the two expression.
        """
        res = deepcopy(self)

        if self.n == other.n:
            res.a_b_expr += other.a_b_expr
        else:
            raise NotImplementedError()
        
        return res


def get_basis_partition(n) -> list[tuple[int]]:
    """
    Generates all integer partitions of n into increasing parts.

    Return:  Every partition of n into a sum of integers in increasing order.
    Example:  3 -> [[1, 1, 1], [1, 2], [3]].
    """
    # TODO implement the decreasing part as well with question.

    def partitions(n, I=1):
        yield (n,)
        for i in range(I, n//2 + 1):
            for p in partitions(n-i, i):
                yield (i,) + p

    return list(partitions(n))

def compute_generic_witt_map(n: int, basis_partition: list[tuple[int]]) -> Expression:
    """
    Constructs an expression as a linear combination of partitions of n, with symbolic coefficents alpha_i.
    """
    a, b = symbols('a b')
    alphas = symbols(' '.join([f'alpha_{i}' for i in range(len(basis_partition))]))
    if n == 1:
        alphas = [alphas]
    generic_witt_map = Expression(sympify('0'), n)
    print(alphas)
    for alpha, partition in zip(alphas, basis_partition):
        mapp = Expression(sympify('1'), 0)
        for i in range(len(partition)):
            expr= Expression(-a-partition[i]*b,partition[i])
            mapp *= expr 
        generic_witt_map = generic_witt_map + mapp * alpha
    
    return generic_witt_map

def get_alpha_coefficients_matrix(n: int, expr: Expression, basis_partition: list[tuple[int]]) -> np.ndarray:
    """
    Builds the matrix M of coefficient for alpha variable, where M[i,j] is the coeffcient of alpha_j
      in the expression for the a**ib**j term.
    
    """
    res = []
    alphas = symbols(' '.join([f'alpha_{i}' for i in range(len(basis_partition))]))
    if n == 1:
        alphas = [alphas]
    a, b = symbols('a b')
    v =[]
    for i in range(n+1):
        for j in range(n+1):
            coeff_ij = expr.a_b_expr.expand().coeff(a, i).coeff(b, j)
            v.append(coeff_ij)

    for eq in v:
        l=[]
        for alpha in alphas:
            l.append(eq.expand().coeff(alpha,1).evalf())
        l.append(eq.expand().coeff(0,0).evalf())
        res.append(l)
    
    return np.array(res, dtype=np.float32)    

def get_expanded_coefficients_vector(n: int, expr: Expression, basis_partition: list[tuple[int]]) -> np.ndarray:
    """
    Extracts the expanded coefficients of an expression for all combinations of powers of a and b.
    """
    a, b = symbols('a b')
    v =[]
    for i in range(n+1):
        for j in range(n+1):
            x = expr.a_b_expr.expand().coeff(a, i).coeff(b, j)
            v.append(x) 
    return np.array(v, dtype=np.float32)


def solve_linear_system(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(M, y)[0]

def result_from_alphas(alphas: np.ndarray, basis_partition: list[Symbol]) -> str:
    """
    Converts the alphas into a string.
    """
    s = ""
    for alpha, partition in zip(alphas, basis_partition):
        if alpha != 0:
            if s == "" and alpha < 0:
                s += "-"
            elif s != "":
                s += " + " if alpha > 0 else " - "

            if not np.allclose(abs(alpha), 1):
                s += str(abs(alpha)) + " * "

            occs = defaultdict(int)
            for i in partition:
                occs[i] += 1
            s += " ".join(f"e_{i}^{k}" if k > 1 else f"e_{i}" for i, k in occs.items())
    
    return s

@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Path to the input file containing the expression."),
    verbose: bool = typer.Option(default=False, help="Whether to print intermediate steps.")
):
    # TODO handle multiple t
    expr = Expression.from_str(input_file.open().read())
    if verbose:
        print(f"Found expression: {expr}")
    basis_partition = get_basis_partition(expr.n)
    expr_with_alphas = compute_generic_witt_map(expr.n, basis_partition)
    if verbose:
        print("Computed expr with alphas:", expr_with_alphas)
    
    M = get_alpha_coefficients_matrix(expr.n, expr_with_alphas, basis_partition)
    if verbose:
        print("Computed coefficients matrix:", M)
    y = get_expanded_coefficients_vector(expr.n, expr, basis_partition)
    if verbose:
        print("Computed results vector:", y)

    alphas = solve_linear_system(M, y)
    if verbose:
        print("Computed alpha results:", alphas)

    if np.all(M @ alphas == y):
        print(result_from_alphas(alphas, basis_partition))
    else:
        print("Element not in image!")

if __name__ == "__main__":
    app()