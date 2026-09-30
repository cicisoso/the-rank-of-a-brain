"""Step 4: is the difference between the sexes larger than the difference between two animals of the same sex?
Three brains (MaleCNS male, FlyWire female, BANC female) matched through FlyWire-namespace type names, and three nerve cords
(MaleCNS male, MANC male, BANC female) matched through MANC-namespace names (MaleCNS mancType; MANC via MaleCNS mancGroup =
MANC group; BANC manc_cell_type). Types present in all three animals are used. For every animal we build the matrix of input
fractions between shared types (whole animal, left hemisphere, right hemisphere) and compare animals by
  (i) relative capture: energy of matrix A retained by the leading k singular modes of matrix B, divided by the energy retained
      by A's own leading k modes (k = 10...400), for within-animal (left vs right), same-sex and different-sex pairs;
 (ii) difference spectra: singular values of A_L - B_L against the within-animal left-right differences;
(iii) entrywise similarity (Pearson correlation of input fractions).
Writes results/sex_individual.json and results/sex_individual_spectra.npz."""
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
KS=[10,25,50,100,200,400]; out={}; spectra={}
for reg,D in REG.items():
    names=list(D); shared=sorted(set.intersection(*[set(pd.Series(v[1]).dropna()) for v in D.values()])); shared=np.array(shared)
    mats={}
    for nm,(W,lab,side,sex) in D.items():
        mats[nm]=type_frac(W,lab,shared); mats[nm+'_L']=type_frac(W,lab,shared,side=='L'); mats[nm+'_R']=type_frac(W,lab,shared,side=='R')
    svd={k:la.svd(v,full_matrices=False) for k,v in mats.items()}
    res={'n_shared':int(len(shared)),'animals':{nm:D[nm][3] for nm in names}}
    for k,v in mats.items():
        s=svd[k][1]; res[f'PR_{k}']=float((s**2).sum()**2/(s**4).sum()); spectra[f'{reg}|{k}']=s
    def rel(x,y): Ux,Sx,Vx=svd[x]; Uy,Sy,Vy=svd[y]; return [captured(mats[x],Uy,Vy,k)/captured(mats[x],Ux,Vx,k) for k in KS]
    cap={}
    for nm in names: cap[f'{nm}_L|{nm}_R']=dict(kind='within animal',rel=rel(nm+'_L',nm+'_R')); cap[f'{nm}_R|{nm}_L']=dict(kind='within animal',rel=rel(nm+'_R',nm+'_L'))
    for x,y in itertools.permutations(names,2):
        kind='same sex' if D[x][3]==D[y][3] else 'different sex'
        cap[f'{x}|{y}']=dict(kind=kind,rel=rel(x,y))
        for s_ in 'LR': cap[f'{x}_{s_}|{y}_{s_}']=dict(kind=kind+' (hemisphere)',rel=rel(f'{x}_{s_}',f'{y}_{s_}'))
    res['capture']=cap; res['k']=KS
    # entrywise similarity between animals and between hemispheres
    corr={}
    for x,y in itertools.combinations(names,2): corr[f'{x}|{y}']=float(np.corrcoef(mats[x].ravel(),mats[y].ravel())[0,1])
    for nm in names: corr[f'{nm}_L|{nm}_R']=float(np.corrcoef(mats[nm+'_L'].ravel(),mats[nm+'_R'].ravel())[0,1])
    res['pearson']=corr
    # difference spectra: same hemisphere across animals vs left-right within animals
    diff={}
    for nm in names: s=la.svdvals(mats[nm+'_L']-mats[nm+'_R']); spectra[f'{reg}|diff|{nm}_L-{nm}_R']=s; diff[f'{nm} L-R']=dict(kind='within animal',energy=float((s**2).sum()),sigma1=float(s[0]))
    noise=max(v['sigma1'] for v in diff.values())
    for x,y in itertools.combinations(names,2):
        for s_ in 'LR':
            s=la.svdvals(mats[f'{x}_{s_}']-mats[f'{y}_{s_}']); spectra[f'{reg}|diff|{x}_{s_}-{y}_{s_}']=s
            diff[f'{x}-{y} {s_}']=dict(kind='same sex' if D[x][3]==D[y][3] else 'different sex',energy=float((s**2).sum()),sigma1=float(s[0]),n_above_LR=int((s>noise).sum()))
    res['diff']=diff; res['noise_sigma1']=noise
    out[reg]=res
    print(reg,'shared',len(shared),{k:round(v,1) for k,v in res.items() if k.startswith('PR_') and '_' not in k[3:]},flush=True)
    for k,v in cap.items(): print('  ',k,v['kind'],[round(x,3) for x in v['rel']])
    print('   pearson',{k:round(v,3) for k,v in corr.items()})
    print('   diff',{k:(v['kind'],round(v['energy'],1),round(v['sigma1'],3),v.get('n_above_LR')) for k,v in diff.items()},flush=True)
json.dump(out,open(f'{R}/sex_individual.json','w'),indent=1); np.savez_compressed(f'{R}/sex_individual_spectra.npz',**spectra); print('done')
