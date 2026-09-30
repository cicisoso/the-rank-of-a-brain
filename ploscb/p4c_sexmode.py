"""Step 4c: the leading mode of the male-female difference among shared brain types. For the difference between the MaleCNS
and each female brain (FlyWire, BANC), same hemisphere, report the types that carry the leading mode, whether the modes from the
two female references agree (|cosine| of singular vectors), and whether the loading types are enriched for fruitless/doublesex
expression or dimorphism annotations (MaleCNS fields). Writes results/sex_mode.json."""
import os, json, numpy as np, pandas as pd, scipy.linalg as la
from scipy import stats
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'p4_sex_individual.py')).read().split('def captured')[0])
D=REG['brain']; shared=np.array(sorted(set.intersection(*[set(pd.Series(v[1]).dropna()) for v in D.values()])))
mats={f'{nm}_{s}':type_frac(W,lab,shared,side==s) for nm,(W,lab,side,sex) in D.items() for s in 'LR'}
fru=mn.assign(ft=mn.flywireType.where(mn.region.isin(['central','optic']))).groupby('ft').fruDsx.apply(lambda v: v.notna().mean()).reindex(shared).fillna(0).values>0
dim=mn.assign(ft=mn.flywireType.where(mn.region.isin(['central','optic']))).groupby('ft').dimorphism.apply(lambda v: v.notna().mean()).reindex(shared).fillna(0).values>0
sc=mn.assign(ft=mn.flywireType.where(mn.region.isin(['central','optic']))).groupby('ft').superclass.agg(lambda v: v.value_counts().idxmax()).reindex(shared).values
out={}
modes={}
for fem in ['FlyWire','BANC']:
    for s in 'LR':
        Dm=mats[f'MaleCNS_{s}']-mats[f'{fem}_{s}']; U,S,Vt=la.svd(Dm,full_matrices=False); modes[(fem,s)]=(U[:,0],Vt[0],S)
        w=U[:,0]**2+Vt[0]**2; top=np.argsort(-w)[:20]; top5=w>=np.quantile(w,0.99)
        o,p=stats.fisher_exact([[int((fru&top5).sum()),int((~fru&top5).sum())],[int((fru&~top5).sum()),int((~fru&~top5).sum())]])
        o2,p2=stats.fisher_exact([[int((dim&top5).sum()),int((~dim&top5).sum())],[int((dim&~top5).sum()),int((~dim&~top5).sum())]])
        out[f'MaleCNS-{fem} {s}']=dict(sigma=[float(x) for x in S[:5]],top_types=[str(shared[i]) for i in top],top_superclass=[str(sc[i]) for i in top],
            energy_share_top20=float(w[top].sum()/w.sum()),fru_top1pct=[int((fru&top5).sum()),int(top5.sum())],fru_odds=float(o),fru_p=float(p),dim_odds=float(o2),dim_p=float(p2),
            superclass_of_top1pct=pd.Series(sc[top5]).value_counts().to_dict())
        print(fem,s,out[f'MaleCNS-{fem} {s}']['top_types'][:12],out[f'MaleCNS-{fem} {s}']['superclass_of_top1pct'],'fru',out[f'MaleCNS-{fem} {s}']['fru_top1pct'],round(p,3),'share',round(w[top].sum()/w.sum(),2),flush=True)
for s in 'LR':
    u1,v1,_=modes[('FlyWire',s)]; u2,v2,_=modes[('BANC',s)]; out[f'agreement {s}']=dict(cos_u=float(abs(u1@u2)),cos_v=float(abs(v1@v2)))
    print('agreement',s,out[f'agreement {s}'])
json.dump(out,open(os.path.join(R,'sex_mode.json'),'w'),indent=1,default=str); print('done')
