import sys; sys.argv=[sys.argv[0],"12","6"]
import check418 as C
from witt_kernel import partitions
P=C.P
N=int(__import__('os').environ.get('NMAX','16'))
psi=C.make_T(2)
def kernel(w):
    monos=sorted(partitions(w),key=lambda m:(len(m),m))
    span={}; ker=[]
    for m in monos:
        img=dict(psi({m:1})); comb={m:1}
        while img:
            piv=max(img)
            if piv in span:
                iv,cv=span[piv]; c=img[piv]*pow(iv[piv],P-2,P)
                for k,v in iv.items(): C.addto(img,k,-c*v)
                for k,v in cv.items(): C.addto(comb,k,-c*v)
            else: break
        if img: span[max(img)]=(img,comb)
        else: ker.append((len(m),comb))
    return ker,len(monos)
def br(k,z):
    r=C.umul(C.gen(k),z)
    for m,c in C.umul(z,C.gen(k)).items(): C.addto(r,m,-c)
    return r
K={}; print(" w | dimU | ker | ker by order | rank Psi on U2 | NF(J=e1..e3) | dimU-ker | new gens (order)")
Kspan={}
for w in range(1,N+1):
    ker,dimU=kernel(w); K[w]=ker
    S=C.Span(); [S.add(v) for _,v in ker]; Kspan[w]=S
    J=C.Span()
    for i in range(1,w):
        for x in Kspan[w-i].rows.values():
            J.add(C.umul(C.gen(i),x)); J.add(C.umul(x,C.gen(i)))
    new={}
    for o,v in sorted(ker,key=lambda t:t[0]):
        if J.add(v): new[o]=new.get(o,0)+1
    byo={}
    for o,_ in ker: byo[o]=byo.get(o,0)+1
    u2=[m for m in partitions(w) if len(m)<=2]
    R=C.Span(); rk=sum(R.add(psi({m:1})) for m in u2)
    nf=sum(1 for m in partitions(w) if sum(1 for i in m if i>=4)<=1)
    print(f"{w:2d} | {dimU:4d} | {len(ker):3d} | {str(dict(sorted(byo.items()))):22s} | {rk:3d} | {nf:4d} | {dimU-len(ker):4d} | {new or '-'}")
# lowest kernel element and differentiator check
low=min(w for w in K if K[w])
print("\nlowest weight:",low, "element:",{m:(c if c<P//2 else c-P) for m,c in K[low][0][1].items()})
for m in (5,6):
    for k in range(m+1, low+2):
        s_=low-k
        if s_>=1 and k-m>=1:
            om=C.omega(m,k,s_)
            if om: print(f"Omega^({m})_({k},{s_}) in ker: {not psi(om)}  =",{mm:(c if c<P//2 else c-P) for mm,c in om.items()})
