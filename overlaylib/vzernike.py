import numpy as np, pandas as pd, time
from .core import Decomposer
from .utils import grad_Z, enum_uv
from .overlay_io import OutputBundle

class VecZernike(Decomposer):
    def fit(self, df: pd.DataFrame):
        t0=time.time()
        modes=enum_uv(self.cfg["n_max"], self.cfg.get("include_B",True))
        K,M=len(df),len(modes)
        A=np.zeros((2*K,M))
        for j,(fam,n,m) in enumerate(modes):
            dx,dy=grad_Z(n,m,df.xn.values,df.yn.values)
            vx,vy=(dx,dy) if fam=="U" else (dy,-dx)
            A[0::2,j]=vx; A[1::2,j]=vy
        b=np.empty(2*K); b[0::2]=df.dx; b[1::2]=df.dy
        coef,*_=np.linalg.lstsq(A,b,rcond=None)
        fit=A@coef; fit_vec=np.column_stack([fit[0::2],fit[1::2]])
        coeff=pd.DataFrame(modes,columns=["family","n","m"]); coeff["coef"]=coef
        rms=np.sqrt(np.mean((df[["dx","dy"]].values-fit_vec)**2))
        return OutputBundle("vz",coeff,fit_vec,rms,{"fit_sec":time.time()-t0})