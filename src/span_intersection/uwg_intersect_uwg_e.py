from collections import defaultdict

BASIS_ELEMENTS_ASCENDING = True

class Term:
    coeff: int
    es: list[int]

    def __init__(self, coeff, es):
        self.coeff = coeff
        self.es = es

    def __mul__(self, other: "Term") -> "Expr":
        coeff = self.coeff * other.coeff
        es = self.es + other.es

        if sorted(es, reverse=not BASIS_ELEMENTS_ASCENDING) == es:
            return Expr([Term(coeff, es)])

        # find index of first out-of-order element
        for i in range(len(es) - 1):
            if BASIS_ELEMENTS_ASCENDING:
                if es[i] > es[i + 1]:
                    break
            else:
                if es[i] < es[i + 1]:
                    break

        # multiply the out-of-order elements
        e1, e2 = es[i], es[i + 1]
        new_coeff = e2 - e1

        # recursively multiply the rest of the elements
        # return es[:i] * e_2*e_1 +- (1-2) e_(1+2) * es[i+2:]
        left_expr = Term(coeff, es[:i]) * Term(1, [e2, e1] + es[i + 2:])
        right_expr = Term(coeff * new_coeff, es[:i] + [e1 + e2]) * Term(1, es[i + 2:])

        return left_expr + right_expr

    def __repr__(self) -> str:
        if self.coeff == 0:
            return "0"

        res = ""
        if self.coeff != 1:
            res += f"{self.coeff} "
        for e in self.es:
            res += f"e_{e} "

        return res.rstrip()

class Expr:
    terms: list[Term]

    def __init__(self, terms = []):
        self.terms = terms

    def __add__(self, other: "Expr") -> "Expr":
        term_dict = defaultdict(int)
        for t in self.terms + other.terms:
            term_dict[tuple(t.es)] += t.coeff

        return Expr([Term(v, list(k)) for k, v in term_dict.items() if v != 0])
    
    def __mul__(self, other: "Expr") -> "Expr":
        if isinstance(other, int):
            return Expr([Term(t.coeff * other, t.es) for t in self.terms])
        
        res = Expr([])
        for t1 in self.terms:
            for t2 in other.terms:
                res += t1 * t2
        
        return res + Expr([])

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        return " + ".join(repr(t) for t in self.terms)

    def __sub__(self, other: "Expr") -> "Expr":
        return self + Expr([Term(-t.coeff, t.es) for t in other.terms])

def e(n: int) -> Expr:
    return Expr([Term(1, [n])])

def g(n: int, m: int) -> Expr:
    return e(n) * e(m) - e(1) * e(n+m-1) + (e(n+m) * (n-1))

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
        res.append(Expr([e(n) for n in p]))
    return res

uw_3 = UW_basis(3)
g_22 = g(2, 2)
uw_2 = UW_basis(2)
g_23 = g(2, 3)

# print(uw_3)
# print(g_22)
# print(uw_2)
# print(g_23)
# print(e(1)*e(1)*e(1)*g(2,2))
# print(e(3)*g(2,2))
print(e(4)*e(3)*e(2)*e(1))