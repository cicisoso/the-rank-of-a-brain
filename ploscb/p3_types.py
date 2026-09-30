"""Step 3: cell-type redundancy in BANC and MANC, as for MaleCNS and FlyWire (type_redundancy.py, types_sex.py):
least-squares block model with cell types or cell types split by side (and random groups of the same sizes), and full spectra
of the signed, input-normalised type-level matrix against its degree-preserving null. Writes results/types_banc_manc.json."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from speclib import frob2, input_normalise, signed, configuration_model
DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); T=5; out={}
def hutch_s4(mv, rmv, n, n_probe=32, seed=0):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n_probe):
        z=rng.choice([-1.0,1.0],size=n); y=rmv(mv(z)); v.append(float(y@y))
    return float(np.mean(v))
def block(M, MT, codes, nt):
    N=M.shape[0]; n_s=np.bincount(codes,minlength=nt).astype(float)
    P=sp.csr_matrix((np.ones(N),(codes,np.arange(N))),shape=(nt,N)); S=(P@M@P.T).tocsr(); Dinv=sp.diags(1.0/n_s); B=(Dinv@S@Dinv).tocsr()
    energy=float((sp.diags(n_s)@B.multiply(B)@sp.diags(n_s)).sum()); f2=frob2(M)
    bmv=lambda z: P.T@(B@(P@z)); brmv=lambda z: P.T@(B.T@(P@z))
    pr_block=energy**2/hutch_s4(bmv,brmv,N); pr_res=(f2-energy)**2/hutch_s4(lambda z: M@z-bmv(z),lambda z: MT@z-brmv(z),N)
    return dict(energy=energy,frac=energy/f2,PR_block=float(pr_block),PR_residual=float(pr_res),n_groups=int(nt))
def full_spectrum(A):
    A=A.toarray() if sp.issparse(A) else A; s=la.svdvals(A); f2=(s**2).sum(); c=np.cumsum(s**2)/f2
    return s, dict(N=A.shape[0],PR=float(f2**2/(s**4).sum()),PR_over_N=float(f2**2/(s**4).sum()/A.shape[0]),stable_rank=float(f2/s[0]**2),**{f'rank_{int(f*100)}':int(np.searchsorted(c,f)+1) for f in [0.5,0.9]})
def type_matrix(M, labels):
    lab=pd.Series(labels); ok=lab.notna().values; idx=np.where(ok)[0]; cats=pd.Categorical(lab[ok]); types=np.array(cats.categories)
    P=sp.csr_matrix((np.ones(len(idx)),(cats.codes,idx)),shape=(len(types),M.shape[0])); return (P@M@P.T).tocsr(), types
spectra={}
for name,nf,mf in [('banc',f'{DA}/banc_nodes.parquet',f'{DA}/banc_cns_w{T}.npz'),('manc',f'{DA}/manc_nodes.parquet',f'{DA}/manc_vnc_w{T}.npz')]:
    nodes=pd.read_parquet(nf); W=sp.load_npz(mf).tocsr()
    for flav in ['count','signed_norm']:
        t0=time.time(); M=W if flav=='count' else input_normalise(signed(W,nodes.sign.values)); MT=M.T.tocsr(); N=M.shape[0]
        pr_full=frob2(M)**2/hutch_s4(lambda z:M@z,lambda z:MT@z,N)
        for grouping in ['type','type_side']:
            lab=nodes.type.copy(); lab=lab.where(lab.notna(),'__u'+pd.Series(np.arange(len(lab))).astype(str))
            if grouping=='type_side': lab=lab+'|'+nodes.somaSide.fillna('M').astype(str)
            cat=pd.Categorical(lab); codes=cat.codes; nt=len(cat.categories)
            r=block(M,MT,codes,nt); rr=block(M,MT,np.random.default_rng(0).permutation(codes),nt)
            out[f'{name}|{flav}|{grouping}']=dict(N=int(N),n_typed=int(nodes.type.notna().sum()),n_types=int(nodes.type.nunique()),PR_full=float(pr_full),groups=r,random_groups=rr)
            print(name,flav,grouping,'PR_full',round(pr_full,1),'frac',round(r['frac'],3),'PR_block',round(r['PR_block'],1),'random frac',round(rr['frac'],4),round(time.time()-t0),'s',flush=True)
    # type-level matrix spectra (signed, input-normalised) against a degree-preserving null
    A,types=type_matrix(W,nodes.type.values)
    ts=pd.Series(nodes.sign.values,index=nodes.type.values).groupby(level=0).mean().reindex(types).fillna(1).values; ts=np.where(ts<0,-1,1)
    B=input_normalise(signed(sp.csr_matrix(A),ts)); s,res=full_spectrum(B)
    Bn=input_normalise(configuration_model(signed(sp.csr_matrix(A),ts),seed=1)); sn,rn=full_spectrum(Bn)
    res['null_config']=rn; res['n_above_edge']=int((s>sn[0]).sum()); out[f'{name}_types|signed_norm']=res; spectra[f'{name}_types|signed_norm']=s; spectra[f'{name}_types|signed_norm|null_config']=sn
    print(name,'type matrix',res,flush=True)
    json.dump(out,open(f'{R}/types_banc_manc.json','w'),indent=1)
# the same type-matrix procedure for MaleCNS (whole CNS) and FlyWire (brain), for a like-for-like comparison of four connectomes
Q=os.path.dirname(HERE)
for name,nf,mf in [('malecns',f'{Q}/data/male_nodes.parquet',f'{Q}/data/male_cns_w{T}.npz'),('flywire',f'{Q}/data/female_nodes.parquet',f'{Q}/data/female_brain_w{T}.npz')]:
    nodes=pd.read_parquet(nf); W=sp.load_npz(mf).tocsr(); A,types=type_matrix(W,nodes.type.values)
    ts=pd.Series(nodes.sign.values,index=nodes.type.values).groupby(level=0).mean().reindex(types).fillna(1).values; ts=np.where(ts<0,-1,1)
    B=input_normalise(signed(sp.csr_matrix(A),ts)); s,res=full_spectrum(B)
    Bn=input_normalise(configuration_model(signed(sp.csr_matrix(A),ts),seed=1)); sn,rn=full_spectrum(Bn)
    res['null_config']=rn; res['n_above_edge']=int((s>sn[0]).sum()); out[f'{name}_types|signed_norm']=res; spectra[f'{name}_types|signed_norm']=s; spectra[f'{name}_types|signed_norm|null_config']=sn
    print(name,'type matrix',res,flush=True); json.dump(out,open(f'{R}/types_banc_manc.json','w'),indent=1)
np.savez_compressed(f'{R}/type_spectra_banc_manc.npz',**spectra); print('done')
