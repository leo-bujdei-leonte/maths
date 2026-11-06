import sys; sys.argv=[sys.argv[0],"2147483629","16"]
import witt_kernel as WK
from math import comb
c=lambda i: (-1)**i*comb(3,i) if 0<=i<=3 else 0
# coefficient identities used in the bracket proofs
print("e1 identity:",all(-r*c(r)+(r-4)*c(r-1)==0 for r in range(-1,6)))
print("e2 identity:",all(-r*c(r)-3*c(r-1)+(r-5)*c(r-2)==0 for r in range(-1,7)))
# K_w = span of D_{w-s,s}: compare dims with kernel dims
ok=True
for w in range(3,15):
    S=WK.Span()
    for s in range(1,w-3): S.add(WK.omega(3,w-s,s))
    kern,_=WK.kernel_by_order(w)
    T=WK.Span()
    for o,v in kern:
        if o<=2: T.add(v)
    same=len(S)==len(T) and all(not T.reduce(v) for v in S.rows.values())
    ok&=same
print("K_w = span{D_{w-s,s}} for 3<=w<=14:",ok)
