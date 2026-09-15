"""Signal rank: number of singular values of each connectome matrix that exceed the largest singular value of its null models
(the 'spikes above the bulk edge'), plus the excess energy carried by those modes. Uses merged spectra."""
import json, numpy as np
S=json.load(open('results/spectrum_summary.json')); Z=np.load('results/spectra.npz'); TZ=np.load('results/type_spectra.npz'); TS=json.load(open('results/types_sex.json'))
out={}
for key in S:
    s=Z[key]; res={}
    for nm in ['null_config','null_wshuf','null_er']:
        k=f'{key}|{nm}'
        if k in Z:
            sn=Z[k]; edge=float(sn[0]); n=int((s>edge).sum()); m=min(len(s),len(sn))
            excess=float(((s[:m]**2-sn[:m]**2).clip(0)).sum()/S[key]['frob2'])
            res[nm]=dict(edge=edge,n_above=n,excess_energy_topk=excess,sigma1_ratio=float(s[0]/sn[0]))
    if res: out[key]=res
for key in [k for k in TZ.files if 'null' not in k and '|' in k]:
    k=f'{key}|null_config'
    if k in TZ.files:
        s=TZ[key]; sn=TZ[k]; edge=float(sn[0]); out[key]={'null_config':dict(edge=edge,n_above=int((s>edge).sum()),excess_energy=float(((s**2-sn**2).clip(0)).sum()/(s**2).sum()),sigma1_ratio=float(s[0]/sn[0]))}
json.dump(out,open('results/signal_rank.json','w'),indent=1)
for k,v in out.items(): print(k,{a:(b['n_above'],round(b['sigma1_ratio'],2),round(b.get('excess_energy_topk',b.get('excess_energy',0)),3)) for a,b in v.items()})
