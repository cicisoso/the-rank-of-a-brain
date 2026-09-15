"""Null models for the signed, input-normalised flavour must be normalised after rewiring (otherwise permuting targets breaks the
column normalisation). Recompute the three nulls for the whole male CNS and the female brain and update the summaries."""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp
from speclib import *
T=5; R='results'; name=sys.argv[1]
nodes=pd.read_parquet('data/male_nodes.parquet' if name=='male_cns' else 'data/female_nodes.parquet')
M=sp.load_npz(f'data/male_cns_w{T}.npz' if name=='male_cns' else f'data/female_brain_w{T}.npz')
Ms=signed(M,nodes.sign.values); real=input_normalise(Ms); f2=frob2(real)
S=json.load(open(f'{R}/spectrum_{name}.json')); Z=dict(np.load(f'{R}/spectra_{name}.npz'))
key=f'{name}|signed_norm'
for nm,fn_ in [('config',configuration_model),('wshuf',shuffle_weights),('er',erdos_renyi)]:
    Nn=input_normalise(fn_(Ms)); r=participation_ratio(Nn,n_probe=32); _,Sn,_,_=topk(Nn,k=800,n_iter=4); cn=energy_curve(Sn,r['frob2'])
    S[key][f'null_{nm}']={'PR':r['PR'],'PR_over_N':r['PR_over_N'],'sigma_max':float(Sn[0]),'stable_rank':r['frob2']/Sn[0]**2,**{f'rank_{int(f*100)}':rank_at(cn,f) for f in [0.5,0.8,0.9]},'energy_topk':float(cn[-1]),'k':int(len(Sn)),'renormalised':True}
    Z[f'{key}|null_{nm}']=Sn; print(name,nm,S[key][f'null_{nm}'],flush=True)
json.dump(S,open(f'{R}/spectrum_{name}.json','w'),indent=1); np.savez_compressed(f'{R}/spectra_{name}.npz',**Z); print('done')
