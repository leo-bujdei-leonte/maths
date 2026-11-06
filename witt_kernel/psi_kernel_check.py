#!/usr/bin/env python3
"""
Check whether an element of U(W_{>=-1}) lies in the kernel of the orbit homomorphism Psi_n.

    Psi_n(e_i) = t^{i+1} d  +  sum_{j=0}^{n-1} binom(i+1, j+1) t^{i-j} v_j,

with target T_n = A_1 (x) U(g_n), where A_1 = k<t, d> (d t - t d = 1), and g_n has basis
v_0, ..., v_{n-1} with [v_i, v_j] = (j - i) v_{i+j} if i + j <= n - 1, and 0 otherwise.
(This is Pham-Timmins' Psi_n; restricting to indices >= 1 gives Psi_n^+ on U(W_{>=1}).)
All arithmetic is exact (fractions).

Usage examples
--------------
  python3 psi_kernel_check.py -n 1 "e5e1 - 3e4e2 + 3e3e3 - e2e4"
  python3 psi_kernel_check.py -n 1 --omega 3 5 1          # differentiator Omega^(3)_{5,1}
  python3 psi_kernel_check.py -n 2 --omega 5 7 1
  python3 psi_kernel_check.py -n 1 "e3e3 + 1/3 e1e5 - 4/3 e2e4 + 2/3 e6"
  python3 psi_kernel_check.py -n 1 "e3e-1 - 4e2e0 + 6e1e1 - 4e0e2 + e-1e3"   # index -1 allowed
  python3 psi_kernel_check.py --selftest

Input syntax
------------
A sum of terms  [coefficient] word, separated by + or -.
  * coefficient: integer or fraction, e.g. 3, -4/3, optionally followed by '*'
  * word: letters e<i> written next to each other, e.g. e5e1, e3*e3, e_{-1}e3, e-1e3
    (a minus sign directly after 'e' is part of the index; elsewhere it is subtraction)
  * a constant term is allowed, e.g. "2" or "1/2"
"""
import argparse
import re
import sys
from fractions import Fraction
from functools import lru_cache
from math import comb


# ---------------------------------------------------------------- parsing
TERM_RE = re.compile(r"\s*([+-])?\s*([0-9]+(?:/[0-9]+)?)?\s*\*?\s*((?:e_?\{?-?[0-9]+\}?\s*\*?\s*)*)")
LETTER_RE = re.compile(r"e_?\{?(-?[0-9]+)\}?")


def parse_element(text):
    """Return {word(tuple of ints): Fraction}."""
    s = text.strip()
    if not s:
        raise ValueError("empty input")
    out = {}
    pos = 0
    while pos < len(s):
        m = TERM_RE.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse near: {s[pos:pos+20]!r}")
        sign, coef, word = m.group(1), m.group(2), m.group(3)
        if coef is None and not word.strip():
            raise ValueError(f"cannot parse near: {s[pos:pos+20]!r}")
        c = Fraction(coef) if coef else Fraction(1)
        if sign == "-":
            c = -c
        letters = tuple(int(x) for x in LETTER_RE.findall(word))
        for i in letters:
            if i < -1:
                raise ValueError(f"index {i} < -1 is not in W_(>=-1)")
        out[letters] = out.get(letters, 0) + c
        pos = m.end()
    return {w: c for w, c in out.items() if c != 0}


def omega(m, k, s):
    """Differentiator Omega^(m)_{k,s} = sum_i (-1)^i binom(m,i) e_{k-i} e_{s+i}."""
    out = {}
    for i in range(m + 1):
        w = (k - i, s + i)
        out[w] = out.get(w, 0) + Fraction((-1) ** i * comb(m, i))
    return {w: c for w, c in out.items() if c != 0}


# ---------------------------------------------------------------- the target T_n
def make_psi(n):
    """Return a function computing Psi_n on {word: coeff}; result is {(a,b,vword): coeff}
    meaning sum coeff * t^a d^b (x) v_{vword}, with vword a nondecreasing tuple (PBW in U(g_n))."""

    @lru_cache(maxsize=None)
    def vnormal(word):
        # PBW normal form in U(g_n): nondecreasing indices
        for i in range(len(word) - 1):
            a, b = word[i], word[i + 1]
            if a > b:
                res = {}
                for m, c in vnormal(word[:i] + (b, a) + word[i + 2:]):
                    res[m] = res.get(m, 0) + c
                if a + b <= n - 1 and b - a != 0:
                    for m, c in vnormal(word[:i] + (a + b,) + word[i + 2:]):
                        res[m] = res.get(m, 0) + (b - a) * c
                return tuple((m, c) for m, c in res.items() if c != 0)
        return ((word, 1),)

    def falling(x, k):
        r = 1
        for j in range(k):
            r *= x - j
        return r

    def tmul(x, y):
        res = {}
        for (a, b, V), cx in x.items():
            for (a2, b2, V2), cy in y.items():
                vprod = vnormal(V + V2)
                # d^b t^a2 = sum_k binom(b,k) a2!/(a2-k)! t^(a2-k) d^(b-k)
                for k in range(min(b, a2) + 1):
                    coef = comb(b, k) * falling(a2, k)
                    for Vm, cv in vprod:
                        key = (a + a2 - k, b - k + b2, Vm)
                        res[key] = res.get(key, 0) + cx * cy * coef * cv
        return {k: c for k, c in res.items() if c != 0}

    def psi_letter(i):
        d = {}
        d[(i + 1, 1, ())] = Fraction(1)              # t^{i+1} d   (i >= -1)
        for j in range(n):
            c = comb(i + 1, j + 1) if i + 1 >= 0 else 0
            if c:
                d[(i - j, 0, (j,))] = Fraction(c)   # binom(i+1,j+1) t^{i-j} v_j
        return d

    @lru_cache(maxsize=None)
    def psi_word(word):
        if not word:
            return (((0, 0, ()), Fraction(1)),)
        prev = dict(psi_word(word[:-1]))
        return tuple(tmul(prev, psi_letter(word[-1])).items())

    def psi(element):
        res = {}
        for w, c in element.items():
            for k, v in psi_word(w):
                res[k] = res.get(k, 0) + c * v
        return {k: c for k, c in res.items() if c != 0}

    return psi


# ---------------------------------------------------------------- printing
def fmt_coef(c, first):
    sgn = "-" if c < 0 else ("" if first else "+")
    a = abs(c)
    s = "" if a == 1 else (str(a) + " ")
    return (sgn + " " if not first or sgn else "") + s


def fmt_element(el):
    if not el:
        return "0"
    parts = []
    for i, (w, c) in enumerate(sorted(el.items(), key=lambda t: (len(t[0]), t[0]))):
        word = "".join(f"e{j}" if j >= 0 else f"e_{{{j}}}" for j in w) or "1"
        if word == "1":
            parts.append((("-" if c < 0 else ("" if i == 0 else "+")) + " " + str(abs(c))).strip())
        else:
            parts.append((fmt_coef(c, i == 0) + word).strip())
    return " ".join(parts)


def fmt_T(img):
    if not img:
        return "0"
    parts = []
    for i, ((a, b, V), c) in enumerate(sorted(img.items())):
        mon = []
        if a: mon.append("t" + (f"^{a}" if a != 1 else ""))
        if b: mon.append("d" + (f"^{b}" if b != 1 else ""))
        for j in V: mon.append(f"v{j}")
        if mon:
            parts.append((fmt_coef(c, i == 0) + "*".join(mon)).strip())
        else:
            parts.append((("-" if c < 0 else ("" if i == 0 else "+")) + " " + str(abs(c))).strip())
    return " ".join(parts)


def weight(el):
    ws = {sum(w) for w in el}
    return ws.pop() if len(ws) == 1 else sorted(ws)


# ---------------------------------------------------------------- main
def check(n, el, verbose=True):
    psi = make_psi(n)
    img = psi(el)
    if verbose:
        print(f"n = {n}")
        print(f"element : {fmt_element(el)}")
        print(f"weight  : {weight(el)}")
        print(f"Psi_n   : {fmt_T(img)}")
        print(f"in ker Psi_{n}: {'YES' if not img else 'NO'}")
    return not img


def selftest():
    cases = [
        (1, omega(3, 5, 1), True, "Omega^(3)_{5,1}, n=1"),
        (1, omega(3, 4, 2), True, "Omega^(3)_{4,2} = -Omega^(3)_{5,1}, n=1"),
        (1, parse_element("e3e3 + 1/3 e1e5 - 4/3 e2e4 + 2/3 e6"), True, "Omega/3 in PBW form, n=1"),
        (1, omega(4, 3, -1), True, "Omega^(4)_{3,-1} (Pham-Timmins generator), n=1"),
        (1, parse_element("e1e5"), False, "e1e5 alone, n=1"),
        (2, omega(5, 7, 1), True, "Omega^(5)_{7,1}, n=2"),
        (2, omega(3, 5, 1), False, "Omega^(3)_{5,1} is NOT in ker Psi_2"),
        (2, omega(6, 5, -1), True, "Omega^(6)_{5,-1} (Pham-Timmins generator), n=2"),
        (1, parse_element("e3e4 + 1/2 e1e6 - 3/2 e2e5 + 1/2 e7"), True, "z_{34}, n=1"),
    ]
    ok = True
    for n, el, expected, name in cases:
        got = check(n, el, verbose=False)
        flag = "ok" if got == expected else "FAILED"
        ok &= got == expected
        print(f"[{flag}] {name}: in kernel = {got}")
    print("all self-tests passed" if ok else "SOME SELF-TESTS FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check whether an element of U(W_{>=-1}) is in ker Psi_n.")
    ap.add_argument("element", nargs="?", help='e.g. "e5e1 - 3e4e2 + 3e3e3 - e2e4"')
    ap.add_argument("-n", type=int, default=1, help="which orbit homomorphism Psi_n (default 1)")
    ap.add_argument("--omega", nargs=3, type=int, metavar=("M", "K", "S"),
                    help="use the differentiator Omega^(M)_{K,S} as the element")
    ap.add_argument("--selftest", action="store_true", help="run built-in checks")
    # An element starting with '-' (e.g. "-e5e1+3e4e2") would be read as an option:
    # protect such tokens by a leading space (parse_element strips it).
    argv = sys.argv[1:]
    argv = [(" " + a) if re.match(r"^-\s*(e|[0-9])", a) and not (k > 0 and argv[k - 1] == "-n") else a
            for k, a in enumerate(argv)]
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.omega:
        el = omega(*args.omega)
    elif args.element:
        try:
            el = parse_element(args.element)
        except ValueError as exc:
            ap.error(str(exc))
    else:
        ap.error("give an element, --omega M K S, or --selftest")
    check(args.n, el)
