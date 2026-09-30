"""Step 5: sex-specific patches in both sexes, with 50 matched random draws.
  male-specific neurons of MaleCNS (dimorphism 'male-specific' or 'potentially male-specific'): brain matrix and whole CNS;
  female-specific neurons of BANC (sexually_dimorphic 'female-specific'): brain (induced) and whole CNS.
Tests, as in male_specific.py but with 50 draws: (i) participation ratio of the edge set touching the patch against random
isomorphic sets matched on superclass and decile of total synapse count; (ii) loss of participation ratio on removing the
patch against matched random removals; (iii) containment of the patch neurons' output and input vectors in the leading
50/200/1000 modes of the isomorphic network (fitted on 90% of isomorphic neurons) against held-out isomorphic neurons.
Usage: python p5_patches.py <case>, case in male_brain, male_cns, female_brain, female_cns."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); sys.path.insert(0,Q)
from speclib import participation_ratio, topk
DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); T=5; NDRAW=50
case=sys.argv[1]; t0=time.time()
if case.startswith('male'):
    nodes=pd.read_parquet(f'{Q}/data/male_nodes.parquet')
    if case=='male_brain': W=sp.load_npz(f'{Q}/data/male_brain_w{T}.npz'); keep=nodes.region.isin(['central','optic']).values
    else: W=sp.load_npz(f'{Q}/data/male_cns_w{T}.npz'); keep=np.ones(len(nodes),bool)
    patch=nodes.dimorphism.isin(['male-specific','potentially male-specific']).values; dim=nodes.dimorphism.isin(['sexually dimorphic','potentially sexually dimorphic']).values
else:
    nodes=pd.read_parquet(f'{DA}/banc_nodes.parquet'); W=sp.load_npz(f'{DA}/banc_cns_w{T}.npz')
    keep=nodes.region.isin(['central','optic']).values if case=='female_brain' else np.ones(len(nodes),bool)
    patch=(nodes.dimorphism=='female-specific').values; dim=(nodes.dimorphism=='dimorphic').values
idx=np.where(keep)[0]; W=W[idx][:,idx].tocsr(); nb=nodes.iloc[idx].reset_index(drop=True); patch=patch[idx]; dim=dim[idx]&~patch; iso=~patch&~dim
deg=np.asarray(W.sum(0)).ravel()+np.asarray(W.sum(1)).ravel()
out=dict(case=case,N=int(len(nb)),n_patch=int(patch.sum()),n_dim=int(dim.sum()),n_iso=int(iso.sum()),syn_frac_patch=float(deg[patch].sum()/deg.sum()),n_draws=NDRAW)
print(out,flush=True)
dec=pd.qcut(np.log1p(deg),10,labels=False,duplicates='drop'); key=(nb.superclass.astype(str)+'|'+pd.Series(dec).astype(str)).values
def matched(mask,n,seed):
    r=np.random.default_rng(seed); need=pd.Series(key[mask]).value_counts(); pool={k:np.where((key==k)&iso)[0] for k in need.index}; draws=[]
    for _ in range(n):
        sel=np.concatenate([r.choice(pool[k],min(c,len(pool[k])),replace=False) for k,c in need.items()]); m=np.zeros(len(nb),bool); m[sel]=True; draws.append(m)
    return draws
def touching(mask): d=sp.diags(mask.astype(float)); return (d@W+W@d-d@W@d).tocsr()
def remove(mask): k=np.where(~mask)[0]; return W[k][:,k].tocsr()
def empirical_p(obs,null,side):   # one-sided, with the +1 correction
    null=np.asarray(null); return float(((null<=obs).sum()+1)/(len(null)+1)) if side=='less' else float(((null>=obs).sum()+1)/(len(null)+1))
# (i) patch rank
r=participation_ratio(touching(patch),n_probe=48); rand=[participation_ratio(touching(m),n_probe=24)['PR'] for m in matched(patch,NDRAW,2)]
out['patch_rank']=dict(PR=r['PR'],rand_mean=float(np.mean(rand)),rand_sd=float(np.std(rand)),rand=rand,p_lower=empirical_p(r['PR'],rand,'less'),PR_per_neuron=r['PR']/patch.sum())
print('patch',{k:v for k,v in out['patch_rank'].items() if k!='rand'},round(time.time()-t0),'s',flush=True)
# (ii) removal
full=participation_ratio(W,n_probe=48)['PR']; rm=participation_ratio(remove(patch),n_probe=48)['PR']
rr=[participation_ratio(remove(m),n_probe=24)['PR'] for m in matched(patch,NDRAW,1)]; d_rand=[full-x for x in rr]
out['removal']=dict(PR_full=full,dPR=full-rm,rand_dPR_mean=float(np.mean(d_rand)),rand_dPR_sd=float(np.std(d_rand)),rand_dPR=d_rand,p_greater=empirical_p(full-rm,d_rand,'greater'))
print('removal',{k:v for k,v in out['removal'].items() if k!='rand_dPR'},round(time.time()-t0),'s',flush=True)
# (iii) containment
rng=np.random.default_rng(0); hold=iso&(rng.random(len(nb))<0.1); tr=np.where(iso&~hold)[0]
U,S,Vt,dt=topk(W[tr][:,tr].tocsr(),k=1000,n_iter=5)
def contain(mask,kind,k):
    ix=np.where(mask)[0]; X=(W[ix][:,tr] if kind=='out' else W[tr][:,ix].T).toarray(); P=X@Vt[:k].T if kind=='out' else X@U[:,:k]
    n2=(X**2).sum(1); ok=n2>0; return (P[ok]**2).sum(1)/n2[ok]
C={}
for k in [50,200,1000]:
    for kind in ['out','in']:
        cp,ch,cd=contain(patch,kind,k),contain(hold,kind,k),contain(dim,kind,k)
        C[f'{kind}_k{k}']=dict(patch_median=float(np.median(cp)),dim_median=float(np.median(cd)) if len(cd) else None,iso_median=float(np.median(ch)),
                               auc=float(stats.mannwhitneyu(cp,ch).statistic/(len(cp)*len(ch))),p=float(stats.mannwhitneyu(cp,ch).pvalue),n_patch=int(len(cp)),n_iso=int(len(ch)))
        if k==200: np.savez_compressed(f'{R}/containment_{case}_{kind}.npz',patch=cp,iso=ch,dim=cd)
        print('contain',kind,k,C[f'{kind}_k{k}'],flush=True)
out['containment']=C
json.dump(out,open(f'{R}/patch_{case}.json','w'),indent=1,default=float); print('done',case,round(time.time()-t0),'s')
