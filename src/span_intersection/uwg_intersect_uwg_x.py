"""
g_22 = e_2^2 - e_1*e_3 + e_4
e_1^3 * g_22
"""
import numpy as np

class Term:
    """A term of the form coeff * x^n * ∂^k."""
    coeff: int
    x_pow: int
    d_pow: int

    def __init__(self, coeff: int, x_pow: int, d_pow: int):
        self.coeff = coeff
        self.x_pow = x_pow
        self.d_pow = d_pow

    def __mul__(self, other: "Term") -> list["Term"]:
        if self.d_pow == 0:
            return [Term(self.coeff * other.coeff, self.x_pow + other.x_pow, other.d_pow)]

        terms_1 = Term(self.coeff * other.coeff, self.x_pow, self.d_pow - 1) * Term(1, other.x_pow, other.d_pow + 1)
        terms_2 = Term(self.coeff * other.coeff * other.x_pow, self.x_pow, self.d_pow - 1) * Term(1, other.x_pow - 1, other.d_pow)

        return terms_1 + terms_2

    def __repr__(self): # TODO 1 0 0 doesn't print
        if self.coeff == 0:
            return "0"

        res = ""
        if self.coeff != 1:
            res += f"{self.coeff} "
        if self.x_pow != 0:
            res += f"x"
            if self.x_pow != 1:
                res += f"^{self.x_pow}"
            res += " "
        if self.d_pow != 0:
            res += "∂"
            if self.d_pow != 1:
                res += f"^{self.d_pow}"

        return res

class Expr:
    """A sum of terms of the form coeff * x^n * ∂^k."""
    terms: list[Term]

    def __init__(self, terms: list[Term] = []):
        self.terms = terms

    def __add__(self, other):
        term_dict = {}
        for t in self.terms + other.terms:
            if (t.x_pow, t.d_pow) in term_dict:
                term_dict[(t.x_pow, t.d_pow)] += t.coeff
            else:
                term_dict[(t.x_pow, t.d_pow)] = t.coeff

        # Remove zero coefficients and sort by x_pow
        return Expr(sorted((Term(c, xp, dp) for (xp, dp), c in term_dict.items() if c != 0), key=lambda t: (-t.x_pow, -t.d_pow)))

    def __sub__(self, other):
        return self + Expr([Term(-t.coeff, t.x_pow, t.d_pow) for t in other.terms])

    def __mul__(self, other):
        if isinstance(other, int):
            return Expr([Term(t.coeff * other, t.x_pow, t.d_pow) for t in self.terms])
        
        res = Expr([Term(0, 0, 0)])
        for t1 in self.terms:
            for t2 in other.terms:
                res += Expr(t1 * t2)
        return res

    def __repr__(self):
        if not self.terms:
            return "0"
        return " + ".join(repr(t) for t in self.terms)

    def __pow__(self, power: int): # overengineered "exponentiation by squaring"
        if power == 0:
            return Expr([Term(1, 0, 0)])
        elif power == 1:
            return self
        else:
            half = self ** (power // 2)
            if power % 2 == 0:
                return half * half
            else:
                return half * half * self
    
def e(n: int, base_coeff: int = -1) -> Expr:
    """TODO figure out if base_coeff is 1 or -1"""
    return Expr([Term(base_coeff, n + 1, 1)])

def g(n: int, m: int, mode: int = 1) -> Expr:
    if mode == 1:
        return e(n) * e(m) - e(n - 1) * e(m + 1) + e(n + m)
    elif mode == 2:
        return e(n) * e(m) - e(1) * e(n+m-1) + (e(n+m) * (n-1))
    else:
        raise ValueError("Invalid mode")

def partitions(n, I=1):
    """
    Returns partitions in increasing order of elements.
    Source: https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
    """
    yield (n,)
    for i in range(I, n//2 + 1):
        for p in partitions(n-i, i):
            yield (i,) + p

def UW_basis(n: int) -> list[Expr]:
    res = []
    for p in list(partitions(n))[::-1]:
        term = Expr([Term(1, 0, 0)])
        for part in p:
            term = term * e(part)
        res.append(term)
    return res

def span_intersection(UW1_basis: list[Expr], g1: Expr, UW2_basis: list[Expr], g2: Expr, verbose: bool = False):
    basis1 = [e * g1 for e in sorted(UW1_basis, key=lambda expr: -len(expr.terms))]
    basis2 = [e * g2 for e in sorted(UW2_basis, key=lambda expr: -len(expr.terms))]

    n = max(t.x_pow for e in basis1 + basis2 for t in e.terms)
    k = max(t.d_pow for e in basis1 + basis2 for t in e.terms)
    assert all(t.x_pow - t.d_pow == n - k for e in basis1 + basis2 for t in e.terms), "Terms must have equal order"

    if verbose:
        print("Basis 1:")
        print(*basis1, sep="\n", end="\n\n")
        print("Basis 2:")
        print(*basis2, sep="\n", end="\n\n")

    m = np.zeros((n+1, len(basis1) + len(basis2)))
    for i in range(len(basis1)):
        for t in basis1[i].terms:
            m[t.x_pow, i] += t.coeff
    for i in range(len(basis2)):
        for t in basis2[i].terms:
            m[t.x_pow, i + len(basis1)] -= t.coeff
    if verbose:
        print("Coefficient matrix:")
        print(m, end="\n\n")

    # solution = np.linalg.lstsq(m.T, np.zeros(len(basis1) + len(basis2)))[0]
    # print(solution)

    _, S, Vt = np.linalg.svd(m)
    tol = 1e-12
    null_mask = (S <= tol)
    null_space = Vt[null_mask, :].T

    if verbose:
        print("Solutions:")
        print(null_space, end="\n\n")


    

span_intersection(
    UW_basis(3),
    g(2, 2),
    UW_basis(2),
    g(2, 3),
    verbose=True,
)