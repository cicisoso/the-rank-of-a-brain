"""How much of the neuron-level rank is cell-type redundancy? The least-squares block model replaces every entry by the mean
over its (presynaptic group, postsynaptic group) pair; it is the orthogonal projection of W onto block-constant matrices and
has rank <= number of groups. Groups are cell types, or cell types split by soma side. Computed without materialising the
neuron-level block matrix: energy = sum n_s n_t B_st^2, singular values equal those of D^1/2 B D^1/2 (D = group sizes), and
the residual acts as a linear operator for the Hutchinson estimate of sum(sigma^4)."""
import json, numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la
from speclib import frob2, input_normalise, signed
T=5; out={}
def hutch_s4(matvec, rmatvec, n, n_probe=32, seed=0):
    rng=np.random.default_rng(seed); v=[]
    for _ in range(n_probe):
        z=rng.choice([-1.0,1.0],size=n); y=rmatvec(matvec(z)); v.append(float(y@y))
    return float(np.mean(v))
def block(M, MT, codes, nt):
    N=M.shape[0]; n_s=np.bincount(codes,minlength=nt).astype(float)
    P=sp.csr_matrix((np.ones(N),(codes,np.arange(N))),shape=(nt,N)); S=(P@M@P.T).tocsr()
    Dinv=sp.diags(1.0/n_s); B=(Dinv@S@Dinv).tocsr()                      # type-pair means (sparse)
    energy=float((sp.diags(n_s)@B.multiply(B)@sp.diags(n_s)).sum())     # sum n_s n_t B_st^2
    f2=frob2(M); res_energy=f2-energy
    # block matrix Wh = P^T B P as an operator; its sum(sigma^4) by Hutchinson
    bmv=lambda z: P.T@(B@(P@z)); brmv=lambda z: P.T@(B.T@(P@z))
    pr_block=energy**2/hutch_s4(bmv,brmv,N)
    mv=lambda z: M@z - bmv(z); rmv=lambda z: MT@z - brmv(z)
    pr_res=res_energy**2/hutch_s4(mv,rmv,N)
    return dict(energy=energy,frac=energy/f2,PR_block=float(pr_block),PR_residual=float(pr_res),n_groups=int(nt))
for sex,nodes_f,mat_f in [('male','data/male_nodes.parquet',f'data/male_cns_w{T}.npz'),('female','data/female_nodes.parquet',f'data/female_brain_w{T}.npz')]:
    nodes=pd.read_parquet(nodes_f); W=sp.load_npz(mat_f).tocsr()
    for flav in ['count','signed_norm']:
        M=W if flav=='count' else input_normalise(signed(W,nodes.sign.values)); MT=M.T.tocsr(); N=M.shape[0]
        pr_full=frob2(M)**2/hutch_s4(lambda z:M@z,lambda z:MT@z,N)
        for grouping in ['type','type_side']:
            lab=nodes.type.copy(); lab=lab.where(lab.notna(),'__u'+pd.Series(np.arange(len(lab))).astype(str))
            if grouping=='type_side': lab=lab+'|'+nodes.somaSide.fillna('M').astype(str)
            cat=pd.Categorical(lab); codes=cat.codes; nt=len(cat.categories)
            r=block(M,MT,codes,nt); rr=block(M,MT,np.random.default_rng(0).permutation(codes),nt)
            out[f'{sex}|{flav}|{grouping}']=dict(N=int(N),n_typed=int(nodes.type.notna().sum()),PR_full=float(pr_full),groups=r,random_groups=rr)
            print(sex,flav,grouping,'PR_full',round(pr_full,1),'groups:',{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()},'random:',{k:(round(v,4) if isinstance(v,float) else v) for k,v in rr.items()},flush=True)
            json.dump(out,open('results/type_redundancy.json','w'),indent=1)
print('done')
