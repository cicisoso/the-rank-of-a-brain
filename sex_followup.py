"""Within-animal reference for the cross-sex subspace comparison: cross-projection energy between the left and right
hemisphere type matrices of each animal and between random split halves, on the same shared types and normalisation."""
import json, numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la
T=5; mn=pd.read_parquet('data/male_nodes.parquet'); fn=pd.read_parquet('data/female_nodes.parquet')
Mbr=sp.load_npz(f'data/male_brain_w{T}.npz'); F=sp.load_npz(f'data/female_brain_w{T}.npz')
mn['ft']=mn.flywireType.where(mn.region.isin(['central','optic'])); shared=np.array(sorted(set(mn.ft.dropna())&set(fn.type.dropna())))
def type_frac(M,labels,mask=None):
    lab=pd.Series(labels); ok=lab.isin(shared).values&(np.ones(len(lab),bool) if mask is None else mask); idx=np.where(ok)[0]
    cat=pd.Categorical(lab[ok],categories=shared); P=sp.csr_matrix((np.ones(len(idx)),(cat.codes,idx)),shape=(len(shared),M.shape[0])); A=(P@M@P.T).toarray(); col=A.sum(0); col[col==0]=1; return A/col
def captured(W,U,V,k): P=U[:,:k]@(U[:,:k].T@W@V[:k].T)@V[:k]; return float((P**2).sum()/(W**2).sum())
ks=[10,25,50,100,200,400,800]; out={'k':ks}
mats={'male_L':type_frac(Mbr,mn.ft.values,(mn.somaSide=='L').values),'male_R':type_frac(Mbr,mn.ft.values,(mn.somaSide=='R').values),'female_L':type_frac(F,fn.type.values,(fn.somaSide=='L').values),'female_R':type_frac(F,fn.type.values,(fn.somaSide=='R').values),'male':type_frac(Mbr,mn.ft.values),'female':type_frac(F,fn.type.values)}
r=np.random.default_rng(1).random(Mbr.shape[0])<0.5; mats['male_A']=type_frac(Mbr,mn.ft.values,r); mats['male_B']=type_frac(Mbr,mn.ft.values,~r)
r=np.random.default_rng(2).random(F.shape[0])<0.5; mats['female_A']=type_frac(F,fn.type.values,r); mats['female_B']=type_frac(F,fn.type.values,~r)
svd={k:la.svd(v,full_matrices=False) for k,v in mats.items()}
pairs=[('male_L','male_R'),('female_L','female_R'),('male_A','male_B'),('female_A','female_B'),('male_L','female_L'),('male_R','female_R'),('male','female'),('female','male')]
for a,b in pairs:
    Ub,Sb,Vbt=svd[b]; Ua,Sa,Vat=svd[a]; out[f'{a}_by_{b}']=[captured(mats[a],Ub,Vbt,k) for k in ks]; out[f'{a}_by_self']=[captured(mats[a],Ua,Vat,k) for k in ks]
    # relative capture = cross / self
    out[f'{a}_by_{b}_rel']=[c/s for c,s in zip(out[f'{a}_by_{b}'],out[f'{a}_by_self'])]
    print(a,'by',b,[round(x,3) for x in out[f'{a}_by_{b}_rel']],flush=True)
# hemisphere-level spectra: PR of L and R and their difference for reference
for k,v in mats.items():
    s=svd[k][1]; out[f'PR_{k}']=float((s**2).sum()**2/(s**4).sum())
json.dump(out,open('results/sex_followup.json','w'),indent=1); print('done')
