"""Step 3b: the cell-type block model of all four connectomes recomputed with 1,024 batched Hutchinson probes (full matrix, block
model and residual), so that the effective ranks in Fig 3B match the precise values of pr_table.json.
Count weights; groups = cell types, cell types split by side, and random groups of the side-resolved sizes."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); sys.path.insert(0,Q)
from speclib import frob2
DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); T=5; NP=1024; out={}; t0=time.time()
def s4(mv,rmv,n,seed=0,batch=64):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(NP//batch):
        Z=rng.choice([-1.0,1.0],size=(n,batch)); Y=rmv(mv(Z)); v.append((Y**2).sum(0))
    v=np.concatenate(v); return float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v)))
def block(M,MT,codes,nt):
    N=M.shape[0]; n_s=np.bincount(codes,minlength=nt).astype(float)
    P=sp.csr_matrix((np.ones(N),(codes,np.arange(N))),shape=(nt,N)); S=(P@M@P.T).tocsr(); Dinv=sp.diags(1.0/n_s); B=(Dinv@S@Dinv).tocsr(); PT=P.T.tocsr(); BT=B.T.tocsr()
    energy=float((sp.diags(n_s)@B.multiply(B)@sp.diags(n_s)).sum()); f2=frob2(M)
    bmv=lambda Z: PT@(B@(P@Z)); brmv=lambda Z: PT@(BT@(P@Z))
    sb,_=s4(bmv,brmv,N); sr,_=s4(lambda Z: M@Z-bmv(Z),lambda Z: MT@Z-brmv(Z),N)
    return dict(frac=energy/f2,PR_block=energy**2/sb,PR_residual=(f2-energy)**2/sr,n_groups=int(nt))
SETS={'MaleCNS':(f'{Q}/data/male_nodes.parquet',f'{Q}/data/male_cns_w{T}.npz'),'FlyWire':(f'{Q}/data/female_nodes.parquet',f'{Q}/data/female_brain_w{T}.npz'),
      'BANC':(f'{DA}/banc_nodes.parquet',f'{DA}/banc_cns_w{T}.npz'),'MANC':(f'{DA}/manc_nodes.parquet',f'{DA}/manc_vnc_w{T}.npz')}
for ds,(nf,mf) in SETS.items():
    nodes=pd.read_parquet(nf); M=sp.load_npz(mf).tocsr(); MT=M.T.tocsr(); N=M.shape[0]
    sf,se=s4(lambda Z:M@Z,lambda Z:MT@Z,N); f2=frob2(M); res=dict(N=int(N),n_typed=int(nodes.type.notna().sum()),n_types=int(nodes.type.nunique()),PR_full=f2**2/sf,PR_full_se=f2**2/sf*se/sf)
    for grouping in ['type','type_side']:
        lab=nodes.type.copy(); lab=lab.where(lab.notna(),'__u'+pd.Series(np.arange(len(lab))).astype(str))
        if grouping=='type_side': lab=lab+'|'+nodes.somaSide.fillna('M').astype(str)
        cat=pd.Categorical(lab); codes=cat.codes; nt=len(cat.categories)
        res[grouping]=block(M,MT,codes,nt)
        if grouping=='type_side': res['random']=block(M,MT,np.random.default_rng(0).permutation(codes),nt)
    out[ds]=res; print(ds,{k:(round(v,3) if isinstance(v,float) else v) for k,v in res.items() if not isinstance(v,dict)},{g:{k:round(v,3) for k,v in res[g].items()} for g in ['type','type_side','random']},round(time.time()-t0),'s',flush=True)
    json.dump(out,open(f'{R}/block_precise.json','w'),indent=1)
print('done')
