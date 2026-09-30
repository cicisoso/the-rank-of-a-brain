"""Step 8: rank scaling with network size and localisation of the leading modes in all four connectomes.
Scaling: participation ratio of induced subnetworks of randomly chosen neurons (3 draws per size) of the MaleCNS CNS, FlyWire
brain, BANC CNS and MANC nerve cord. Localisation: for each of the leading 200 modes, the number of neurons carrying its output
and input loadings (participation ratio of squared loadings) and the share of its energy in each region.
Writes results/scaling_modes.json."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); sys.path.insert(0,Q)
from speclib import participation_ratio
DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); T=5; out={'scaling':{},'modes':{}}; t0=time.time()
NET={'MaleCNS':(sp.load_npz(f'{Q}/data/male_cns_w{T}.npz'),pd.read_parquet(f'{Q}/data/male_nodes.parquet'),f'{Q}/results/modes_male_cns_count.npz'),
     'FlyWire':(sp.load_npz(f'{Q}/data/female_brain_w{T}.npz'),pd.read_parquet(f'{Q}/data/female_nodes.parquet'),f'{Q}/results/modes_female_brain_count.npz'),
     'BANC':(sp.load_npz(f'{DA}/banc_cns_w{T}.npz'),pd.read_parquet(f'{DA}/banc_nodes.parquet'),f'{R}/modes_banc_cns_count.npz'),
     'MANC':(sp.load_npz(f'{DA}/manc_vnc_w{T}.npz'),pd.read_parquet(f'{DA}/manc_nodes.parquet'),f'{R}/modes_manc_all_count.npz')}
for name,(W,nodes,mf) in NET.items():
    W=W.tocsr(); N=W.shape[0]; rows=[]
    for n in [2500,5000,10000,20000,40000,80000,120000,N]:
        if n>N: continue
        prs=[]
        for s in range(3 if n<N else 1):
            idx=np.sort(np.random.default_rng(s).choice(N,n,replace=False)) if n<N else np.arange(N); prs.append(participation_ratio(W[idx][:,idx].tocsr(),n_probe=24,seed=s)['PR'])
        rows.append(dict(N=int(n),PR_mean=float(np.mean(prs)),PR_sd=float(np.std(prs))))
    x=np.log([r['N'] for r in rows]); y=np.log([r['PR_mean'] for r in rows]); slope=float(np.polyfit(x,y,1)[0])
    out['scaling'][name]=dict(rows=rows,slope=slope); print(name,'scaling slope',round(slope,3),[(r['N'],round(r['PR_mean'],1)) for r in rows],round(time.time()-t0),'s',flush=True)
    if os.path.exists(mf):
        z=np.load(mf); U=z['U']; Vt=z['Vt']; idx=z['idx'] if 'idx' in z.files else np.arange(U.shape[0]); reg=nodes.region.values[idx]
        pr_u=(U**2).sum(0)**2/(U**4).sum(0); pr_v=(Vt**2).sum(1)**2/(Vt**4).sum(1)
        cats=list(pd.unique(reg)); share=[]
        for a in range(min(100,U.shape[1])):
            e=U[:,a]**2+Vt[a]**2; share.append([float(e[reg==c].sum()/e.sum()) for c in cats])
        share=np.array(share); single=float((share.max(1)>0.8).mean())
        out['modes'][name]=dict(neurons_out_median_top200=float(np.median(pr_u)),neurons_in_median_top200=float(np.median(pr_v)),neurons_out_median_top20=float(np.median(pr_u[:20])),
                                neurons_in_median_top20=float(np.median(pr_v[:20])),regions=[str(c) for c in cats],share_top100=share.tolist(),frac_modes_single_region=single,
                                neurons_out=pr_u.tolist(),neurons_in=pr_v.tolist())
        print(name,'modes: median neurons out/in (top200)',round(np.median(pr_u)),round(np.median(pr_v)),'single-region share',round(single,2),flush=True)
    json.dump(out,open(f'{R}/scaling_modes.json','w'),indent=1)
print('done')
