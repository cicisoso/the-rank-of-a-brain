"""Step 4b: entrywise similarity (Pearson r of input fractions between shared types) for hemisphere pairs within and between
animals, and the reliability-corrected similarity r_xy / sqrt(r_xx r_yy) (r_xx = left-right correlation of animal x). Same
matrices as p4_sex_individual.py. Writes results/similarity.json."""

import os, sys, json, itertools, numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); T=5
D=os.environ.get('FLYBRAIN_DATA',os.path.join(os.path.dirname(HERE),'data','raw'))
mn=pd.read_parquet(f'{Q}/data/male_nodes.parquet'); fn=pd.read_parquet(f'{Q}/data/female_nodes.parquet')
bn=pd.read_parquet(f'{DA}/banc_nodes.parquet'); an=pd.read_parquet(f'{DA}/manc_nodes.parquet')
a=pd.read_feather(f'{D}/body-annotations-male-cns-v1.0-minconf-0.5.feather',columns=['bodyId','mancType'])
mn=mn.merge(a,on='bodyId',how='left')
Mbr=sp.load_npz(f'{Q}/data/male_brain_w{T}.npz'); Mvnc=sp.load_npz(f'{Q}/data/male_vnc_w{T}.npz'); F=sp.load_npz(f'{Q}/data/female_brain_w{T}.npz')
B=sp.load_npz(f'{DA}/banc_cns_w{T}.npz'); A=sp.load_npz(f'{DA}/manc_vnc_w{T}.npz')
def induced(W,mask):
    d=sp.diags(mask.astype(float)); return (d@W@d).tocsr()
ASC={'ascending','ascending_visceral_circulatory','sensory_ascending'}
banc_brain=induced(B,(bn.region.isin(['central','optic'])|bn.superclass.isin(ASC)).values)
banc_vnc=induced(B,((bn.region=='vnc')|bn.superclass.isin({'descending','sensory_descending'})).values)
REG={'brain':{'MaleCNS':(Mbr,mn.flywireType.where(mn.region.isin(['central','optic'])).values,mn.somaSide.values,'male'),
              'FlyWire':(F,fn.type.values,fn.somaSide.values,'female'),
              'BANC':(banc_brain,bn.fafb_cell_type.values,bn.somaSide.values,'female')},
     'vnc':{'MaleCNS':(Mvnc,mn.mancType.values,mn.somaSide.values,'male'),
            'MANC':(A,an.mancType.values,an.somaSide.values,'male'),
            'BANC':(banc_vnc,bn.manc_cell_type.values,bn.somaSide.values,'female')}}
def type_frac(W,labels,shared,mask=None):
    lab=pd.Series(labels); ok=lab.isin(shared).values&(np.ones(len(lab),bool) if mask is None else mask); idx=np.where(ok)[0]
    cat=pd.Categorical(lab[ok],categories=shared); P=sp.csr_matrix((np.ones(len(idx)),(cat.codes,idx)),shape=(len(shared),W.shape[0]))
    X=(P@W@P.T).toarray(); col=X.sum(0); col[col==0]=1; return X/col
def captured(X,U,Vt,k): P=U[:,:k]@(U[:,:k].T@X@Vt[:k].T)@Vt[:k]; return float((P**2).sum()/(X**2).sum())
out={}
for reg,D in REG.items():
    names=list(D); shared=np.array(sorted(set.intersection(*[set(pd.Series(v[1]).dropna()) for v in D.values()])))
    mats={}
    for nm,(W,lab,side,sex) in D.items():
        mats[nm+'_L']=type_frac(W,lab,shared,side=='L'); mats[nm+'_R']=type_frac(W,lab,shared,side=='R')
    r=lambda a,b: float(np.corrcoef(mats[a].ravel(),mats[b].ravel())[0,1])
    within={nm:r(nm+'_L',nm+'_R') for nm in names}; rows=[]
    for nm in names: rows.append(dict(pair=f'{nm} L vs R',kind='within animal',r=within[nm],r_corrected=None))
    for x,y in itertools.combinations(names,2):
        kind='same sex' if D[x][3]==D[y][3] else 'different sex'
        for s_ in 'LR':
            rv=r(f'{x}_{s_}',f'{y}_{s_}'); rows.append(dict(pair=f'{x} vs {y} ({s_})',kind=kind,r=rv,r_corrected=rv/np.sqrt(within[x]*within[y])))
    out[reg]=dict(n_shared=int(len(shared)),within=within,rows=rows)
    for row in rows: print(reg,row)
json.dump(out,open(f'{R}/similarity.json','w'),indent=1); print('done')
