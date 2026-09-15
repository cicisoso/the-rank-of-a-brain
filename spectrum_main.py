"""Singular value spectra, participation ratios and energy curves for the male CNS, its regions, and the female brain,
with degree-preserving and weight-shuffled null models. Writes results/spectra.npz and results/spectrum_summary.json."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
from speclib import *
T=5; R='results'; K_MAIN=2000; K_SUB=1200; K_NULL=800
ONLY=sys.argv[1] if len(sys.argv)>1 else None
mn=pd.read_parquet('data/male_nodes.parquet'); fn=pd.read_parquet('data/female_nodes.parquet')
Mcns=sp.load_npz(f'data/male_cns_w{T}.npz'); Mbr=sp.load_npz(f'data/male_brain_w{T}.npz'); Mvnc=sp.load_npz(f'data/male_vnc_w{T}.npz'); F=sp.load_npz(f'data/female_brain_w{T}.npz')
def sub(M,mask): idx=np.where(mask)[0]; return M[idx][:,idx].tocsr(), idx
mats={}
mats['male_cns']=(Mcns,mn.sign.values,np.arange(len(mn)))
mats['male_brain']=sub(Mbr,mn.region.isin(['central','optic']).values)+(None,)
mats['male_vnc']=sub(Mvnc,(mn.region=='vnc').values)+(None,)
mats['male_central']=sub(Mbr,(mn.region=='central').values)+(None,)
mats['male_optic']=sub(Mbr,(mn.region=='optic').values)+(None,)
mats['female_brain']=(F,fn.sign.values,np.arange(len(fn)))
mats['female_central']=sub(F,(fn.region=='central').values)+(None,)
mats['female_optic']=sub(F,(fn.region=='optic').values)+(None,)
# fix tuple layout: (M, sign, idx)
for k,v in list(mats.items()):
    if v[2] is None: M,idx,_=v; nodes=fn if k.startswith('female') else mn; mats[k]=(M,nodes.sign.values[idx],idx)
summary={}; spectra={}
def analyse(name,M,k,flavour,nulls=False):
    t0=time.time(); N=M.shape[0]; res=participation_ratio(M,n_probe=48); res['N']=N; res['nnz']=M.nnz; res['sigma_max']=None
    U,S,Vt,dt=topk(M,k=k,n_iter=4); res['sigma_max']=float(S[0]); res['stable_rank']=res['frob2']/S[0]**2
    c=energy_curve(S,res['frob2']); res['energy_topk']=float(c[-1]); res['k']=k
    for f in [0.5,0.8,0.9,0.95]: res[f'rank_{int(f*100)}']=rank_at(c,f)
    res['seconds']=round(time.time()-t0+dt,1); spectra[f'{name}|{flavour}']=S
    if nulls:
        for nm,fn_ in [('config',configuration_model),('wshuf',shuffle_weights),('er',erdos_renyi)]:
            Nn=fn_(M); r=participation_ratio(Nn,n_probe=32); _,Sn,_,_=topk(Nn,k=min(k,K_NULL),n_iter=4); cn=energy_curve(Sn,r['frob2'])
            res[f'null_{nm}']={'PR':r['PR'],'PR_over_N':r['PR_over_N'],'sigma_max':float(Sn[0]),'stable_rank':r['frob2']/Sn[0]**2,**{f'rank_{int(f*100)}':rank_at(cn,f) for f in [0.5,0.8,0.9]},'energy_topk':float(cn[-1]),'k':int(len(Sn))}
            spectra[f'{name}|{flavour}|null_{nm}']=Sn
    summary[f'{name}|{flavour}']=res; print(name,flavour,{a:b for a,b in res.items() if not isinstance(b,dict)},flush=True)
    return U,S,Vt
for name,(M,sign,idx) in mats.items():
    if ONLY and name!=ONLY: continue
    k=K_MAIN if name in ('male_cns','female_brain') else K_SUB
    nulls=name in ('male_cns','female_brain','male_brain','male_central','male_optic','male_vnc','female_central','female_optic')
    U,S,Vt=analyse(name,M,k,'count',nulls=nulls)
    if name in ('male_cns','female_brain','male_brain'):
        np.savez_compressed(f'{R}/modes_{name}_count.npz',U=U[:,:200].astype(np.float32),S=S,Vt=Vt[:200].astype(np.float32),idx=idx)
    Ms=input_normalise(signed(M,sign)); U,S,Vt=analyse(name,Ms,k,'signed_norm',nulls=name in ('male_cns','female_brain'))
    if name in ('male_cns','female_brain','male_brain'):
        np.savez_compressed(f'{R}/modes_{name}_signed_norm.npz',U=U[:,:200].astype(np.float32),S=S,Vt=Vt[:200].astype(np.float32),idx=idx)
    json.dump(summary,open(f'{R}/spectrum_{name}.json','w'),indent=1); np.savez_compressed(f'{R}/spectra_{name}.npz',**spectra)
print('done')
