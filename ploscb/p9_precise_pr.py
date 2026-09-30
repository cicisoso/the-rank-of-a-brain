"""Step 9: precise participation ratios for every network of the four connectomes (count and signed, input-normalised weights),
with 1,024 Rademacher probes (batched) for the Hutchinson estimate of sum(sigma^4) and its standard error, and the same for one
degree-preserving null per network (renormalised after rewiring for the signed flavour). The standard error of PR follows from
that of sum(sigma^4) (the energy sum(sigma^2) is exact). Writes results/pr_table.json."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); sys.path.insert(0,Q)
from speclib import frob2, input_normalise, signed, configuration_model
DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); T=5; NPROBE=1024; t0=time.time()
def pr_precise(M,n_probe=NPROBE,batch=64,seed=0):
    rng=np.random.default_rng(seed); MT=M.T.tocsr(); vals=[]
    for b in range(n_probe//batch):
        Z=rng.choice([-1.0,1.0],size=(M.shape[1],batch)); Y=MT@(M@Z); vals.append((Y**2).sum(0))
    v=np.concatenate(vals); s4=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); f2=frob2(M); pr=f2**2/s4
    return dict(N=int(M.shape[0]),nnz=int(M.nnz),PR=float(pr),PR_se=float(pr*se/s4),PR_over_N=float(pr/M.shape[0]),rel_se=float(se/s4))
def sub(W,mask): idx=np.where(mask)[0]; return W[idx][:,idx].tocsr(), idx
mn=pd.read_parquet(f'{Q}/data/male_nodes.parquet'); fn=pd.read_parquet(f'{Q}/data/female_nodes.parquet')
bn=pd.read_parquet(f'{DA}/banc_nodes.parquet'); an=pd.read_parquet(f'{DA}/manc_nodes.parquet')
Mcns=sp.load_npz(f'{Q}/data/male_cns_w{T}.npz'); Mbr=sp.load_npz(f'{Q}/data/male_brain_w{T}.npz'); Mvnc=sp.load_npz(f'{Q}/data/male_vnc_w{T}.npz'); F=sp.load_npz(f'{Q}/data/female_brain_w{T}.npz')
B=sp.load_npz(f'{DA}/banc_cns_w{T}.npz'); A=sp.load_npz(f'{DA}/manc_vnc_w{T}.npz')
ASC_B={'ascending','ascending_visceral_circulatory','sensory_ascending'}; ASC_M={'ascending neuron','sensory ascending','efferent ascending','descending neuron'}
NETS={('MaleCNS','CNS'):(Mcns,np.arange(len(mn)),mn),('MaleCNS','brain'):sub(Mbr,mn.region.isin(['central','optic']).values)+(mn,),
      ('MaleCNS','central brain'):sub(Mbr,(mn.region=='central').values)+(mn,),('MaleCNS','optic lobes'):sub(Mbr,(mn.region=='optic').values)+(mn,),
      ('MaleCNS','nerve cord'):sub(Mvnc,(mn.region=='vnc').values)+(mn,),
      ('FlyWire','brain'):(F,np.arange(len(fn)),fn),('FlyWire','central brain'):sub(F,(fn.region=='central').values)+(fn,),('FlyWire','optic lobes'):sub(F,(fn.region=='optic').values)+(fn,),
      ('BANC','CNS'):(B,np.arange(len(bn)),bn),('BANC','brain'):sub(B,bn.region.isin(['central','optic']).values)+(bn,),
      ('BANC','central brain'):sub(B,(bn.region=='central').values)+(bn,),('BANC','optic lobes'):sub(B,(bn.region=='optic').values)+(bn,),
      ('BANC','nerve cord'):sub(B,((bn.region=='vnc')&~bn.superclass.isin(ASC_B)).values)+(bn,),
      ('MANC','nerve cord (all traced)'):(A,np.arange(len(an)),an),('MANC','nerve cord'):sub(A,~an['class'].astype(str).isin(ASC_M).values)+(an,)}
out=[]
for (ds,reg),(W,idx,nodes) in NETS.items():
    W=W.tocsr()
    for flav in ['count','signed_norm']:
        if flav=='count': M=W; Nn=configuration_model(W,seed=1)
        else: Ms=signed(W,nodes.sign.values[idx]); M=input_normalise(Ms); Nn=input_normalise(configuration_model(Ms,seed=1))
        r=pr_precise(M); n=pr_precise(Nn,n_probe=256)
        row=dict(dataset=ds,region=reg,flavour=flav,**r,PR_null=n['PR'],PR_null_se=n['PR_se'],null_over_real=n['PR']/r['PR'])
        out.append(row); print(ds,reg,flav,{k:(round(v,4) if isinstance(v,float) else v) for k,v in row.items() if k not in ('dataset','region','flavour')},round(time.time()-t0),'s',flush=True)
        json.dump(out,open(f'{R}/pr_table.json','w'),indent=1)
print('done')
