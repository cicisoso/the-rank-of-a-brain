"""Spectral tools for large sparse connectivity matrices: exact Frobenius energy, Hutchinson estimate of sum(sigma^4),
participation ratio, stable rank, top-k singular values by randomized SVD, and null models."""
import numpy as np, scipy.sparse as sp, time
from sklearn.utils.extmath import randomized_svd
def frob2(M): return float((M.multiply(M)).sum()) if sp.issparse(M) else float((M*M).sum())
def sum_s4(M, n_probe=64, seed=0):
    """sum of sigma^4 = tr((M^T M)^2) = E||M^T M z||^2 for Rademacher z (Hutchinson); returns mean and s.e.m."""
    rng=np.random.default_rng(seed); n=M.shape[1]; MT=M.T.tocsr() if sp.issparse(M) else M.T; vals=[]
    for _ in range(n_probe):
        z=rng.choice([-1.0,1.0],size=n); y=MT@(M@z); vals.append(float(y@y))
    v=np.array(vals); return v.mean(), v.std(ddof=1)/np.sqrt(n_probe)
def participation_ratio(M, n_probe=64, seed=0):
    f2=frob2(M); s4,se=sum_s4(M,n_probe,seed); pr=f2**2/s4; return dict(frob2=f2, sum_s4=s4, sum_s4_sem=se, PR=pr, PR_over_N=pr/min(M.shape))
def topk(M, k=2000, n_iter=6, seed=0, oversamples=100):
    t0=time.time(); U,S,Vt=randomized_svd(M, n_components=k, n_iter=n_iter, n_oversamples=oversamples, random_state=seed, power_iteration_normalizer='QR')
    return U,S,Vt,time.time()-t0
def energy_curve(S, f2):
    c=np.cumsum(S**2)/f2; return c
def rank_at(c, frac):
    i=np.searchsorted(c, frac); return int(i+1) if i<len(c) else None
def input_normalise(M):
    """divide each column (postsynaptic neuron) by its total input so inputs sum to 1 (fraction of input)."""
    col=np.asarray(np.abs(M).sum(0)).ravel(); col[col==0]=1; return (M@sp.diags(1.0/col)).tocsr()
def signed(M, sign):
    """multiply each row (presynaptic neuron) by its neurotransmitter sign."""
    return (sp.diags(sign.astype(float))@M).tocsr()
# ---- null models (preserve the edge multiset; merge duplicates)
def shuffle_weights(M, seed=0):
    rng=np.random.default_rng(seed); C=M.tocoo(); w=C.data.copy(); rng.shuffle(w); return sp.csr_matrix((w,(C.row,C.col)),shape=M.shape)
def configuration_model(M, seed=0):
    """directed configuration model: keep each presynaptic neuron's out-edges (with weights) and permute the target endpoints,
    which preserves every out-degree, the in-degree multiset and the weight multiset."""
    rng=np.random.default_rng(seed); C=M.tocoo(); col=C.col.copy(); rng.shuffle(col); N=sp.csr_matrix((C.data,(C.row,col)),shape=M.shape); N.sum_duplicates(); return N
def erdos_renyi(M, seed=0):
    rng=np.random.default_rng(seed); C=M.tocoo(); n=M.shape[0]; r=rng.integers(0,n,C.nnz); c=rng.integers(0,n,C.nnz); N=sp.csr_matrix((C.data,(r,c)),shape=M.shape); N.sum_duplicates(); return N
def quarter_circle_energy(N, k):
    """cumulative energy fraction of the top-k singular values of an N x N i.i.d. matrix (Marchenko-Pastur, square case):
    singular values s in [0,2] with density (1/pi) sqrt(4 - s^2); E[s^2]=1, PR/N=0.5."""
    s=np.linspace(0,2,20001); dens=np.sqrt(np.clip(4-s**2,0,None))/np.pi; cdf=np.cumsum(dens[::-1])*(s[1]-s[0]); e=np.cumsum((s[::-1]**2*dens[::-1]))*(s[1]-s[0])
    ranks=cdf*N; return np.interp(np.arange(1,k+1),ranks,e)
