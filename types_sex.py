"""Type-level connectivity: full spectra of male and female type x type matrices, shared-type sex comparison
(alignment of singular subspaces, cross-projection), and the rank of the sex difference relative to left-right and split-half noise."""
import json, time, numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la
from speclib import *
T=5; R='results'; rng=np.random.default_rng(0)
mn=pd.read_parquet('data/male_nodes.parquet'); fn=pd.read_parquet('data/female_nodes.parquet')
Mcns=sp.load_npz(f'data/male_cns_w{T}.npz'); Mbr=sp.load_npz(f'data/male_brain_w{T}.npz'); F=sp.load_npz(f'data/female_brain_w{T}.npz')
out={}
def type_matrix(M, labels, mask=None):
    """sum neuron-level counts into a type x type matrix; labels: array of type names (None excluded)."""
    lab=pd.Series(labels); ok=lab.notna().values if mask is None else (lab.notna().values&mask)
    idx=np.where(ok)[0]; cats=pd.Categorical(lab[ok]); codes=cats.codes; types=np.array(cats.categories)
    P=sp.csr_matrix((np.ones(len(idx)),(codes,idx)),shape=(len(types),M.shape[0]))
    return (P@M@P.T).tocsr(), types
def full_spectrum(A):
    A=A.toarray() if sp.issparse(A) else A; s=la.svdvals(A); f2=(s**2).sum(); p=s**2/f2; pr=f2**2/(s**4).sum()
    q=s/s.sum(); q=q[q>0]; erank=float(np.exp(-(q*np.log(q)).sum()))
    c=np.cumsum(s**2)/f2
    return s, dict(N=A.shape[0],PR=float(pr),PR_over_N=float(pr/A.shape[0]),erank=erank,stable_rank=float(f2/s[0]**2),**{f'rank_{int(f*100)}':int(np.searchsorted(c,f)+1) for f in [0.5,0.8,0.9,0.95,0.99]})
def norm_signed(A, sign):
    A=sp.csr_matrix(A); return input_normalise(signed(A,sign))
# ---------------- 1. full type-level spectra (male CNS types; male brain types; female brain types)
spectra={}
for name,M,nodes,lab in [('male_cns_types',Mcns,mn,mn.type.values),('male_brain_types',Mbr,mn,np.where(mn.region.isin(['central','optic']),mn.type,None)),('female_brain_types',F,fn,fn.type.values)]:
    t0=time.time(); A,types=type_matrix(M,lab)
    tsign=pd.Series(nodes.sign.values,index=nodes.type.values if name!='male_brain_types' else lab); tsign=tsign.groupby(level=0).mean().reindex(types).fillna(1).values; tsign=np.where(tsign<0,-1,1)
    for flav,B in [('count',A),('signed_norm',norm_signed(A,tsign))]:
        s,res=full_spectrum(B); res['seconds']=round(time.time()-t0,1); out[f'{name}|{flav}']=res; spectra[f'{name}|{flav}']=s
        # config-model null at type level (degree preserving on the type graph)
        Cn=configuration_model(sp.csr_matrix(B),seed=1); sn,rn=full_spectrum(Cn); out[f'{name}|{flav}']['null_config']=rn; spectra[f'{name}|{flav}|null_config']=sn
        print(name,flav,res,flush=True)
# ---------------- 2. shared types between sexes (male flywireType <-> female primary_type), brain scope
mn['ft']=mn.flywireType.where(mn.region.isin(['central','optic']))
shared=np.array(sorted(set(mn.ft.dropna())&set(fn.type.dropna()))); print('shared types',len(shared),flush=True)
def type_frac(M,labels,shared,mask=None):
    lab=pd.Series(labels); ok=lab.isin(shared).values&(np.ones(len(lab),bool) if mask is None else mask); idx=np.where(ok)[0]
    cat=pd.Categorical(lab[ok],categories=shared); P=sp.csr_matrix((np.ones(len(idx)),(cat.codes,idx)),shape=(len(shared),M.shape[0]))
    A=(P@M@P.T).toarray(); return A
def frac_norm(A):  # fraction of each postsynaptic type's input (within the shared-type matrix)
    col=A.sum(0); col[col==0]=1; return A/col
Am=type_frac(Mbr,mn.ft.values,shared); Af=type_frac(F,fn.type.values,shared)
Wm=frac_norm(Am); Wf=frac_norm(Af)
Um,Sm,Vmt=la.svd(Wm,full_matrices=False); Uf,Sf,Vft=la.svd(Wf,full_matrices=False)
for nm,S,W in [('male',Sm,Wm),('female',Sf,Wf)]:
    f2=(S**2).sum(); c=np.cumsum(S**2)/f2; out[f'shared_{nm}']=dict(N=len(shared),PR=float(f2**2/(S**4).sum()),PR_over_N=float(f2**2/(S**4).sum()/len(shared)),**{f'rank_{int(f*100)}':int(np.searchsorted(c,f)+1) for f in [0.5,0.8,0.9,0.95]})
spectra['shared_male']=Sm; spectra['shared_female']=Sf
# alignment of singular subspaces and cross-projection energy
K=200; O_left=np.abs(Um[:,:K].T@Uf[:,:K]); O_right=np.abs(Vmt[:K]@Vft[:K].T); np.savez_compressed(f'{R}/alignment.npz',O_left=O_left,O_right=O_right,Sm=Sm,Sf=Sf)
def captured(W,U,V,k):  # fraction of ||W||^2 captured by projecting rows onto span(V_k) and columns onto span(U_k)
    P=U[:,:k]@(U[:,:k].T@W@V[:k].T)@V[:k]; return float((P**2).sum()/(W**2).sum())
ks=[10,25,50,100,200,400,800]; cross={'k':ks,'female_by_male':[captured(Wf,Um,Vmt,k) for k in ks],'female_by_female':[captured(Wf,Uf,Vft,k) for k in ks],'male_by_female':[captured(Wm,Uf,Vft,k) for k in ks],'male_by_male':[captured(Wm,Um,Vmt,k) for k in ks]}
# null for alignment: random rotation (i.i.d. subspace of same dimension)
Q=la.qr(rng.standard_normal((len(shared),K)),mode='economic')[0]; Qv=la.qr(rng.standard_normal((len(shared),K)),mode='economic')[0]
cross['female_by_random']=[captured(Wf,Q,Qv.T,k) for k in ks]; out['cross_projection']=cross
print('cross',cross,flush=True)
# ---------------- 3. rank of the sex difference vs left-right and split-half noise
def side_mats(M,labels,sides,shared):
    return [frac_norm(type_frac(M,labels,shared,mask=(sides==s))) for s in ('L','R')]
mL,mR=side_mats(Mbr,mn.ft.values,mn.somaSide.values,shared); fL,fR=side_mats(F,fn.type.values,fn.somaSide.values,shared)
def split_half(M,labels,shared,seed):
    r=np.random.default_rng(seed).random(M.shape[0])<0.5; return frac_norm(type_frac(M,labels,shared,mask=r)), frac_norm(type_frac(M,labels,shared,mask=~r))
mA,mB=split_half(Mbr,mn.ft.values,shared,1); fA,fB=split_half(F,fn.type.values,shared,2)
diffs={'sex':Wm-Wf,'male_LR':mL-mR,'female_LR':fL-fR,'male_split':mA-mB,'female_split':fA-fB,'male_L_vs_female_L':mL-fL}
dspec={k:la.svdvals(v) for k,v in diffs.items()}
for k,s in dspec.items(): spectra[f'diff_{k}']=s; f2=(s**2).sum(); out[f'diff_{k}']=dict(energy=float(f2),PR=float(f2**2/(s**4).sum()),sigma1=float(s[0]),sigma10=float(s[9]))
noise=max(dspec['male_LR'][0],dspec['female_LR'][0],dspec['male_split'][0],dspec['female_split'][0])
out['sex_modes_above_noise']=int((dspec['sex']>noise).sum()); out['noise_sigma1']=float(noise)
out['sex_modes_above_LR']=int((dspec['sex']>max(dspec['male_LR'][0],dspec['female_LR'][0])).sum())
# energy of the sex difference explained by its top modes, and its top loading types with fru/dsx labels
Ud,Sd,Vdt=la.svd(diffs['sex'],full_matrices=False); c=np.cumsum(Sd**2)/(Sd**2).sum(); out['diff_sex_rank_50_80_90']=[int(np.searchsorted(c,f)+1) for f in (0.5,0.8,0.9)]
fru=mn.groupby('ft').fruDsx.apply(lambda v: v.notna().mean()).reindex(shared).fillna(0).values
dim=mn.groupby('ft').dimorphism.apply(lambda v: v.isin(['sexually dimorphic','potentially sexually dimorphic']).mean()).reindex(shared).fillna(0).values
loads=[]
for a in range(10):
    w=Ud[:,a]**2+Vdt[a]**2; top=np.argsort(-w)[:20]
    loads.append(dict(mode=a+1,sigma=float(Sd[a]),top_types=[str(shared[i]) for i in top],fru_frac_top20=float((fru[top]>0).mean()),dim_frac_top20=float((dim[top]>0).mean())))
out['diff_sex_top_modes']=loads; out['fru_frac_all_shared']=float((fru>0).mean()); out['dim_frac_all_shared']=float((dim>0).mean())
# enrichment of fru/dsx and dimorphic types among high-loading types (top 5% by summed loading over the first 10 modes)
w10=(Ud[:,:10]**2).sum(1)+(Vdt[:10]**2).sum(0); top=w10>=np.quantile(w10,0.95)
from scipy import stats
for nm,vec in [('fru',fru>0),('dim',dim>0)]:
    tab=[[int((vec&top).sum()),int((~vec&top).sum())],[int((vec&~top).sum()),int((~vec&~top).sum())]]; o,p=stats.fisher_exact(tab); out[f'enrich_{nm}_top5pct']=dict(table=tab,odds=float(o),p=float(p))
np.savez_compressed(f'{R}/type_spectra.npz',**spectra); json.dump(out,open(f'{R}/types_sex.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in out.items() if not isinstance(v,dict) or 'PR' in v},indent=0,default=float)[:3000]); print('done')
