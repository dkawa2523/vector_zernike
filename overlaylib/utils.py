import numpy as np
from math import factorial

def _R(n,m,r):
    m=abs(m); R=np.zeros_like(r)
    for k in range((n-m)//2+1):
        c=(-1)**k*factorial(n-k)/(factorial(k)*factorial((n+m)//2-k)*factorial((n-m)//2-k))
        R+=c*r**(n-2*k)
    return R
def _dR(n,m,r):
    m=abs(m); d=np.zeros_like(r)
    for k in range((n-m)//2+1):
        p=n-2*k
        if p==0: continue
        c=(-1)**k*factorial(n-k)/(factorial(k)*factorial((n+m)//2-k)*factorial((n-m)//2-k))
        d+=c*p*r**(p-1)
    return d

def grad_Z(n,m,x,y):
    r=np.hypot(x,y); th=np.arctan2(y,x)
    mask=r>1e-14
    dZdx=np.zeros_like(r); dZdy=np.zeros_like(r)
    R=_R(n,abs(m),r[mask]); dR=_dR(n,abs(m),r[mask])
    if m>=0:
        c=np.cos(m*th[mask]); s=np.sin(m*th[mask])
        dR_dr=dR*c; dZ_dth=-m*R*s
    else:
        m2=-m; s=np.sin(m2*th[mask]); c=np.cos(m2*th[mask])
        dR_dr=dR*s; dZ_dth=m2*R*c
    x_m,y_m,r_m=x[mask],y[mask],r[mask]
    dZdx[mask]=dR_dr*(x_m/r_m)+dZ_dth*(-y_m/r_m**2)
    dZdy[mask]=dR_dr*(y_m/r_m)+dZ_dth*( x_m/r_m**2)
    return dZdx,dZdy

def enum_uv(n_max, include_b=True):
    modes=[]
    for n in range(1,n_max+1):
        for m in range(-n,n+1,2):
            modes.append(("U",n,m))
            if include_b: modes.append(("V",n,m))
    return modes