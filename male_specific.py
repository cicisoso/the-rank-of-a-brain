"""Do male-specific neurons add rank? Neuron-level tests on the male brain matrix: (A) participation-ratio change on removal
versus matched random removal; (B) containment of male-specific neurons' input/output vectors in the isomorphic subspace versus
held-out isomorphic neurons; (C) effective rank of the edge set touching male-specific neurons versus matched random sets.
Also: rank scaling with network size and superclass composition of the leading modes of the whole CNS."""
import json, time, numpy as np, pandas as pd, scipy.sparse as sp
from speclib import *
from scipy import stats
T=5; R='results'; rng=np.random.default_rng(0)
mn=pd.read_parquet('data/male_nodes.parquet'); Mbr=sp.load_npz(f'data/male_brain_w{T}.npz'); Mcns=sp.load_npz(f'data/male_cns_w{T}.npz')
br=mn.region.isin(['central','optic']).values; bidx=np.where(br)[0]; W=Mbr[bidx][:,bidx].tocsr(); nb=mn.iloc[bidx].reset_index(drop=True)
ms=nb.dimorphism.isin(['male-specific','potentially male-specific']).values; dm=nb.dimorphism.isin(['sexually dimorphic','potentially sexually dimorphic']).values; iso=~ms&~dm
deg=np.asarray(W.sum(0)).ravel()+np.asarray(W.sum(1)).ravel()
out={'n_brain':int(len(nb)),'n_ms':int(ms.sum()),'n_dim':int(dm.sum()),'n_iso':int(iso.sum()),'syn_ms':float(deg[ms].sum()),'syn_total':float(deg.sum())}
print(out,flush=True)
def matched_random(mask, n_draw, seed):
    """random isomorphic sets matched to `mask` on superclass and log synapse-count decile."""
    r=np.random.default_rng(seed); dec=pd.qcut(np.log1p(deg),10,labels=False,duplicates='drop'); key=nb.superclass.astype(str)+'|'+dec.astype(str)
    pool={k:np.where((key==k).values&iso)[0] for k in key[mask].unique()}; need=key[mask].value_counts(); draws=[]
    for _ in range(n_draw):
        sel=np.concatenate([r.choice(pool[k],min(n,len(pool[k])),replace=False) for k,n in need.items()]); m=np.zeros(len(nb),bool); m[sel]=True; draws.append(m)
    return draws
def remove(Wm,mask): keep=np.where(~mask)[0]; return Wm[keep][:,keep].tocsr()
# ---------------- A. rank change on removal
full=participation_ratio(W,n_probe=48); out['A_full']=full
for nm,mask in [('ms',ms),('ms_dim',ms|dm)]:
    r=participation_ratio(remove(W,mask),n_probe=48); rand=[participation_ratio(remove(W,m),n_probe=24)['PR'] for m in matched_random(mask,10,1)]
    out[f'A_remove_{nm}']=dict(PR=r['PR'],dPR=full['PR']-r['PR'],rand_dPR_mean=float(full['PR']-np.mean(rand)),rand_dPR_sd=float(np.std(rand)),z=float((full['PR']-r['PR']-(full['PR']-np.mean(rand)))/np.std(rand)),n_removed=int(mask.sum()))
    print('A',nm,out[f'A_remove_{nm}'],flush=True)
# ---------------- B. subspace containment
hold=iso&(rng.random(len(nb))<0.1); train=iso&~hold; tr=np.where(train)[0]
Wtr=W[tr][:,tr].tocsr(); K=1000; U,S,Vt,dt=topk(Wtr,k=K,n_iter=5); print('iso subspace fit',dt,'s',flush=True)
def contain(rows_mask, kind, k):
    idx=np.where(rows_mask)[0]
    if kind=='out': X=W[idx][:,tr].toarray(); P=X@Vt[:k].T; 
    else: X=W[tr][:,idx].toarray().T; P=X@U[:,:k]
    n2=(X**2).sum(1); ok=n2>0; return (P[ok]**2).sum(1)/n2[ok], n2[ok]
B={}
for k in [50,200,1000]:
    for kind in ['out','in']:
        cm,nm_=contain(ms,kind,k); ch,nh=contain(hold,kind,k); cd,nd=contain(dm,kind,k)
        # degree-matched comparison: weight held-out neurons to the male-specific synapse-count distribution
        bins=np.quantile(np.log1p(np.r_[nm_,nh]),np.linspace(0,1,11)); bm=np.clip(np.digitize(np.log1p(nm_),bins)-1,0,9); bh=np.clip(np.digitize(np.log1p(nh),bins)-1,0,9)
        wts=np.array([ (bm==b).mean()/max((bh==b).mean(),1e-9) for b in range(10)])[bh]; matched_median=float(np.quantile(np.repeat(ch,np.maximum(1,np.round(wts*10).astype(int))),0.5))
        auc=stats.mannwhitneyu(cm,ch).statistic/(len(cm)*len(ch))
        B[f'{kind}_k{k}']=dict(ms_median=float(np.median(cm)),dim_median=float(np.median(cd)),iso_heldout_median=float(np.median(ch)),iso_heldout_matched_median=matched_median,auc_ms_vs_iso=float(auc),n_ms=int(len(cm)),n_iso=int(len(ch)))
        print('B',kind,k,B[f'{kind}_k{k}'],flush=True)
        if k==200: np.savez_compressed(f'{R}/containment_{kind}.npz',ms=cm,iso=ch,dim=cd,ms_syn=nm_,iso_syn=nh)
out['B_containment']=B
# ---------------- C. effective rank of the edge set touching male-specific neurons
def touching(mask): d=sp.diags(mask.astype(float)); E=d@W+W@d-d@W@d; return E.tocsr()
Ems=touching(ms); r=participation_ratio(Ems,n_probe=48); rand=[participation_ratio(touching(m),n_probe=24)['PR'] for m in matched_random(ms,10,2)]
out['C_touching']=dict(PR_ms=r['PR'],energy_ms=r['frob2'],rand_PR_mean=float(np.mean(rand)),rand_PR_sd=float(np.std(rand)),n_ms=int(ms.sum()),PR_per_neuron=r['PR']/ms.sum())
print('C',out['C_touching'],flush=True)
# ---------------- D. rank scaling with network size (random neuron subsets of the whole CNS)
sc=[]
for n in [5000,10000,20000,40000,80000,120000,len(mn)]:
    prs=[]
    for s in range(3 if n<len(mn) else 1):
        idx=np.sort(np.random.default_rng(s).choice(len(mn),n,replace=False)) if n<len(mn) else np.arange(len(mn)); S_=Mcns[idx][:,idx].tocsr(); prs.append(participation_ratio(S_,n_probe=24)['PR'])
    sc.append(dict(N=int(n),PR_mean=float(np.mean(prs)),PR_sd=float(np.std(prs)))); print('D',sc[-1],flush=True)
out['D_scaling']=sc
# ---------------- E. superclass / region composition of the leading modes of the whole CNS
z=np.load(f'{R}/modes_male_cns_count.npz'); U=z['U']; Vt=z['Vt']; S=z['S']
np.savez_compressed(f'{R}/mode_localisation.npz',pr_left=(U**2).sum(0)**2/(U**4).sum(0),pr_right=(Vt**2).sum(1)**2/(Vt**4).sum(1),S=S[:200])
reg=mn.region.values; sc_=mn.superclass.values; comp={}
for lab_name,lab in [('region',reg),('superclass',sc_)]:
    cats=pd.unique(lab); mat=np.zeros((min(100,U.shape[1]),len(cats)))
    for a in range(mat.shape[0]):
        e=U[:,a]**2+Vt[a]**2
        for j,c in enumerate(cats): mat[a,j]=e[lab==c].sum()/e.sum()
    comp[lab_name]=dict(categories=[str(c) for c in cats],matrix=mat.tolist())
out['E_mode_composition']=comp
json.dump(out,open(f'{R}/male_specific.json','w'),indent=1,default=float); print('done')
