"""Step 2: spectra, participation ratios and null models for the BANC and MANC networks, computed exactly as for MaleCNS and
FlyWire (spectrum_main.py): Hutchinson estimate of sum(sigma^4) with 48 probes, leading singular values by randomized SVD,
degree-preserving (configuration), weight-shuffled and Erdos-Renyi nulls; signed, input-normalised flavour with nulls
renormalised after rewiring. Usage: python p2_spectra.py <network>."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(HERE))
from speclib import *
R=os.path.join(HERE,'results'); DA=os.path.join(HERE,'data'); T=5
ASC_B={'ascending','ascending_visceral_circulatory','sensory_ascending'}; ASC_M={'ascending neuron','sensory ascending','efferent ascending','descending neuron'}
def nets():
    bn=pd.read_parquet(f'{DA}/banc_nodes.parquet'); B=sp.load_npz(f'{DA}/banc_cns_w{T}.npz')
    mn=pd.read_parquet(f'{DA}/manc_nodes.parquet'); M=sp.load_npz(f'{DA}/manc_vnc_w{T}.npz')
    def sub(W,mask): idx=np.where(mask)[0]; return W[idx][:,idx].tocsr(), idx
    out={'banc_cns':(B,np.arange(len(bn)),bn),
         'banc_brain':sub(B,bn.region.isin(['central','optic']).values)+(bn,),
         'banc_central':sub(B,(bn.region=='central').values)+(bn,),
         'banc_optic':sub(B,(bn.region=='optic').values)+(bn,),
         'banc_vnc':sub(B,((bn.region=='vnc')&~bn.superclass.isin(ASC_B)).values)+(bn,),
         'manc_all':(M,np.arange(len(mn)),mn),
         'manc_vnc':sub(M,~mn['class'].astype(str).isin(ASC_M).values)+(mn,)}
    return out
def analyse(M,k,nulls,renorm_sign=None):
    res=participation_ratio(M,n_probe=48); res['N']=M.shape[0]; res['nnz']=int(M.nnz)
    U,S,Vt,dt=topk(M,k=k,n_iter=4); res['sigma_max']=float(S[0]); res['stable_rank']=res['frob2']/S[0]**2
    c=energy_curve(S,res['frob2']); res['energy_topk']=float(c[-1]); res['k']=k
    for f in [0.5,0.8,0.9,0.95]: res[f'rank_{int(f*100)}']=rank_at(c,f)
    spec={'real':S}
    if nulls:
        for nm,fn_ in [('config',configuration_model),('wshuf',shuffle_weights),('er',erdos_renyi)]:
            Nn=fn_(M if renorm_sign is None else renorm_sign)
            if renorm_sign is not None: Nn=input_normalise(Nn)
            r=participation_ratio(Nn,n_probe=32); _,Sn,_,_=topk(Nn,k=min(k,800),n_iter=4); cn=energy_curve(Sn,r['frob2'])
            res[f'null_{nm}']={'PR':r['PR'],'PR_over_N':r['PR_over_N'],'sigma_max':float(Sn[0]),'stable_rank':r['frob2']/Sn[0]**2,'n_above_edge':int((S>Sn[0]).sum()),
                               **{f'rank_{int(f*100)}':rank_at(cn,f) for f in [0.5,0.8,0.9]},'energy_topk':float(cn[-1]),'k':int(len(Sn))}
            spec[nm]=Sn
    return res,spec,U,Vt
if __name__=='__main__':
    name=sys.argv[1]; W,idx,nodes=nets()[name]; t0=time.time(); summ={}; Z={}
    k=2000 if name=='banc_cns' else 1200
    res,spec,U,Vt=analyse(W,k,True); summ[f'{name}|count']=res; Z.update({f'{name}|count|{a}':b for a,b in spec.items()})
    print(name,'count',{a:(round(b,4) if isinstance(b,float) else b) for a,b in res.items() if not isinstance(b,dict)},round(time.time()-t0),'s',flush=True)
    if name in ('banc_cns','banc_brain','manc_vnc','manc_all'):
        np.savez_compressed(f'{R}/modes_{name}_count.npz',U=U[:,:200].astype(np.float32),S=spec['real'],Vt=Vt[:200].astype(np.float32),idx=idx)
        Ms=signed(W,nodes.sign.values[idx]); Mn=input_normalise(Ms)
        res,spec,_,_=analyse(Mn,k,True,renorm_sign=Ms); summ[f'{name}|signed_norm']=res; Z.update({f'{name}|signed_norm|{a}':b for a,b in spec.items()})
        print(name,'signed_norm',{a:(round(b,4) if isinstance(b,float) else b) for a,b in res.items() if not isinstance(b,dict)},round(time.time()-t0),'s',flush=True)
    json.dump(summ,open(f'{R}/spectrum_{name}.json','w'),indent=1); np.savez_compressed(f'{R}/spectra_{name}.npz',**Z); print('done',name,round(time.time()-t0),'s')
