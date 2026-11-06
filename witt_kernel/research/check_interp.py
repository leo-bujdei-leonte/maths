import sys; sys.argv=[sys.argv[0],"2147483629","16"]
from fractions import Fraction
import witt_kernel as WK
P=WK.P
def inv(a): return pow(a%P,P-2,P)
def L(c,b,nodes):
    num=1; den=1
    for d in nodes:
        if d!=c: num*=(b-d); den*=(c-d)
    return num*inv(den)%P
ok=True
for w in range(6,17):
    nodes=[1,2,w-1]
    for b in range(3,w-2):
        a=w-b
        z={}
        for m,cf in WK.umul(WK.gen(a),WK.gen(b)).items(): WK.addto(z,m,cf)
        for c in nodes:
            for m,cf in WK.umul(WK.gen(w-c),WK.gen(c)).items(): WK.addto(z,m,-L(c,b,nodes)*cf)
        if WK.psi(z): ok=False; print("fails",a,b)
print("W>=1: interpolation elements z_ab lie in ker Psi_1 (6<=w<=16):",ok)
