import numpy as np, pandas as pd
from .core import Decomposer
from .overlay_io import OutputBundle

# ------------------------------------------------------------
# 基本 1st-order
P8  = ["Tx","Ty","Rot","Mag","dMagX","dMagY","OrthX","OrthY"]
P10 = P8 + ["BowX","BowY"]                     # r²

# 2nd・3rd・4th-order 追加関数 -------------------------------
def quad_terms():     # QX,QY,Kx,Ky  (x²−y², xy)
    return ["QX","QY","Kx","Ky"]
def clover_terms():   # CloverX,CloverY = r²x, r²y
    return ["CloverX","CloverY"]
def rot_skew3():      # Rot3,Skew3 (3rd rotational/shear)
    return ["Rot3","Skew3"]
def cubic_terms():    # x³,x²y,xy²,y³ for dx/dy → 8 パラメータ
    mons=[(3,0),(2,1),(1,2),(0,3)]
    return [f"x{p}y{q}_X" for p,q in mons]+[f"x{p}y{q}_Y" for p,q in mons]
def quartic_radial(): # Bow2X,Bow2Y = r⁴
    return ["Bow2X","Bow2Y"]

# ------------------------------------------------------------
def build_param_list(n):
    if n==8 : return P8
    if n==10: return P10
    if n==14: return P10 + quad_terms()
    if n==16: return build_param_list(14) + clover_terms()
    if n==18: return build_param_list(16) + rot_skew3()
    if n==26: return build_param_list(18) + cubic_terms()
    if n==30: return build_param_list(26) + quartic_radial()
    raise ValueError("model_terms must be 8/10/14/16/18/26/30")

# 位置→行列寄与 --------------------------------------------------
def design(x,y,p):
    r2=x*x+y*y
    base={
        "Tx":(1,0),"Ty":(0,1),
        "Rot":(y,-x),
        "Mag":(x,y),
        "dMagX":(x,0),"dMagY":(0,y),
        "OrthX":(0,x),"OrthY":(y,0),
        "BowX":(r2,0),"BowY":(0,r2),
        "QX":(x*x-y*y,0),"QY":(0,x*x-y*y),
        "Kx":(x*y,0),"Ky":(0,x*y),
        "CloverX":(r2*x,0),"CloverY":(0,r2*y),
        "Rot3":(y*r2,-x*r2),
        "Skew3":(-x*(x*x-y*y), y*(x*x-y*y)),
        "Bow2X":(r2*r2,0),"Bow2Y":(0,r2*r2),
    }
    if p in base: return base[p]
    # cubic generic monomials
    if "_X" in p:
        pw=p.replace("_X","")[1:]     # drop leading x
        pwr=list(map(int,pw.split("y")))
        return ( (x**pwr[0])*(y**pwr[1]), 0 )
    if "_Y" in p:
        pw=p.replace("_Y","")[1:]
        pwr=list(map(int,pw.split("y")))
        return (0, (x**pwr[0])*(y**pwr[1]) )
    raise KeyError(p)

def solve_poly(df, params):
    K,M=len(df),len(params)
    A=np.zeros((2*K,M))
    for k,(x,y) in enumerate(zip(df.x,df.y)):
        for j,p in enumerate(params):
            dx,dy=design(x,y,p)
            A[2*k,j]=dx; A[2*k+1,j]=dy
    b=np.empty(2*K); b[0::2]=df.dx; b[1::2]=df.dy
    coef,*_=np.linalg.lstsq(A,b,rcond=None)
    fit=A@coef; fit_vec=np.column_stack([fit[0::2],fit[1::2]])
    coeff=pd.DataFrame({"param":params,"coef":coef})
    return coeff,fit_vec

class PolyOverlay(Decomposer):
    def fit(self, df: pd.DataFrame):
        bundles=[]
        for n in self.cfg["model_terms"]:
            params=build_param_list(n)
            coeff,fit_vec=solve_poly(df,params)
            rms=np.sqrt(np.mean((df[["dx","dy"]].values-fit_vec)**2))
            bundles.append(OutputBundle(f"poly{n}",coeff,fit_vec,rms,{}))
        return bundles