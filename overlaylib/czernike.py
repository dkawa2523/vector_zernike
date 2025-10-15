import numpy as np, pandas as pd, time
from .core import Decomposer
from .utils import grad_Z
from .overlay_io import OutputBundle

def complex_fit(df,n_max):
    modes=[(n,m) for n in range(1,n_max+1) for m in range(-n,n+1,2)]
    K,M=len(df),len(modes)
    Ax=np.zeros((K,M)); Ay=np.zeros((K,M))
    for j,(n,m) in enumerate(modes):
        dx,dy=grad_Z(n,m,df.xn.values,df.yn.values)
        Ax[:,j]=dx; Ay[:,j]=dy
    b_re,b_im=df.dx.values,df.dy.values
    A=np.block([[Ax,-Ay],[Ay,Ax]]); b=np.concatenate([b_re,b_im])
    coef,*_=np.linalg.lstsq(A,b,rcond=None)
    cre,cim=coef[:M],coef[M:]
    fit_re=Ax@cre - Ay@cim
    fit_im=Ax@cim + Ay@cre
    coeff=pd.DataFrame(modes,columns=["n","m"])
    coeff["Re"],coeff["Im"]=cre,cim
    fit_vec=np.column_stack([fit_re,fit_im])
    return coeff,fit_vec

class CpxZernike(Decomposer):
    def fit(self,df):
        t0=time.time()
        coeff,fit_vec=complex_fit(df,self.cfg["n_max"])
        rms=np.sqrt(np.mean((df[["dx","dy"]].values-fit_vec)**2))
        return OutputBundle("cz",coeff,fit_vec,rms,{"fit_sec":time.time()-t0})