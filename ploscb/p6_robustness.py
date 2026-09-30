"""Step 6: robustness and estimator validation.
A. Connection threshold and weight transform: participation ratio of each whole connectome and of its degree-preserving null
   for pair thresholds of 1, 3, 5 and 10 synapses (synapse counts), and at threshold 5 for binary and log(1 + count) weights.
B. Estimator validation: on random induced subnetworks of 8,000 neurons, the Hutchinson participation ratio (48 probes) and the
   randomized-SVD leading singular values (k = 200) against exact dense SVD.
Writes results/robustness.json."""
import os, sys, json, time, numpy as np, pandas as pd, scipy.sparse as sp, scipy.linalg as la, pyarrow.feather as pf, pyarrow.compute as pc
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); sys.path.insert(0,Q)
from speclib import participation_ratio, configuration_model, topk
D=os.environ.get('FLYBRAIN_DATA',os.path.join(os.path.dirname(HERE),'data','raw')); DA=os.path.join(HERE,'data'); R=os.path.join(HERE,'results'); out={'A':{},'B':{}}; t0=time.time()
def edges(name):
    if name=='MaleCNS':
        nodes=pd.read_parquet(f'{Q}/data/male_nodes.parquet'); ids=nodes.bodyId.values
        w=pf.read_table(f'{D}/connectome-weights-male-cns-v1.0-minconf-0.5.feather',columns=['body_pre','body_post','weight']).to_pandas(); return w.body_pre.values,w.body_post.values,w.weight.values,ids
    if name=='FlyWire':
        nodes=pd.read_parquet(f'{Q}/data/female_nodes.parquet'); ids=nodes.bodyId.values
        e=pd.read_csv(f'{D}/flywire/connections.csv.gz',usecols=['pre_root_id','post_root_id','syn_count']).groupby(['pre_root_id','post_root_id'],sort=False).syn_count.sum().reset_index()
        return e.pre_root_id.values,e.post_root_id.values,e.syn_count.values,ids
    if name=='BANC':
        nodes=pd.read_parquet(f'{DA}/banc_nodes.parquet'); ids=nodes.bodyId.values
        e=pf.read_table(f'{D}/banc/banc_888_edgelist_simple_v3.feather',columns=['pre','post','count']).to_pandas(); return e.pre.values,e.post.values,e['count'].values,ids
    if name=='MANC':
        nodes=pd.read_parquet(f'{DA}/manc_nodes.parquet'); ids=nodes.bodyId.values
        c=pd.read_csv(f'{D}/manc/traced-connections.csv'); return c.bodyId_pre.values,c.bodyId_post.values,c.weight.values,ids
def csr(pre,post,w,ids):
    idmap=pd.Series(np.arange(len(ids)),index=ids); pi=idmap.reindex(pre).values; qi=idmap.reindex(post).values; m=~(np.isnan(pi)|np.isnan(qi))
    M=sp.csr_matrix((w[m].astype(np.float64),(pi[m].astype(np.int64),qi[m].astype(np.int64))),shape=(len(ids),len(ids))); M.sum_duplicates(); return M
for name in ['MANC','BANC','FlyWire','MaleCNS']:
    pre,post,w,ids=edges(name); res={}
    for tag,thr,fn in [('count>=1',1,None),('count>=3',3,None),('count>=5',5,None),('count>=10',10,None),('binary>=5',5,'bin'),('log>=5',5,'log')]:
        m=w>=thr; W=csr(pre[m],post[m],w[m] if fn is None else (np.ones(m.sum()) if fn=='bin' else np.log1p(w[m])),ids)
        r=participation_ratio(W,n_probe=32); rn=participation_ratio(configuration_model(W,seed=1),n_probe=32)
        res[tag]=dict(nnz=int(W.nnz),PR=r['PR'],PR_over_N=r['PR_over_N'],PR_null=rn['PR'],null_over_real=rn['PR']/r['PR'])
        print(name,tag,{k:round(v,4) if isinstance(v,float) else v for k,v in res[tag].items()},round(time.time()-t0),'s',flush=True)
    out['A'][name]=res; json.dump(out,open(f'{R}/robustness.json','w'),indent=1)
    del pre,post,w
# ---------------- B. estimator validation
mats={'MaleCNS central':(sp.load_npz(f'{Q}/data/male_cns_w5.npz'),pd.read_parquet(f'{Q}/data/male_nodes.parquet').region.eq('central').values),
      'FlyWire central':(sp.load_npz(f'{Q}/data/female_brain_w5.npz'),pd.read_parquet(f'{Q}/data/female_nodes.parquet').region.eq('central').values),
      'MANC':(sp.load_npz(f'{DA}/manc_vnc_w5.npz'),np.ones(sp.load_npz(f'{DA}/manc_vnc_w5.npz').shape[0],bool))}
for name,(W,mask) in mats.items():
    rows=[]
    for seed in range(3):
        idx=np.sort(np.random.default_rng(seed).choice(np.where(mask)[0],8000,replace=False)); S_=W[idx][:,idx].tocsr()
        s=la.svdvals(S_.toarray()); pr_exact=float((s**2).sum()**2/(s**4).sum())
        est=[participation_ratio(S_,n_probe=48,seed=q)['PR'] for q in range(5)]
        _,Sr,_,_=topk(S_,k=200,n_iter=4); relerr=np.abs(Sr-s[:200])/s[:200]
        rows.append(dict(seed=seed,PR_exact=pr_exact,PR_hutch_mean=float(np.mean(est)),PR_hutch_sd=float(np.std(est)),rel_err_PR=float(np.mean(est)/pr_exact-1),
                         rsvd_max_rel_err_top200=float(relerr.max()),rsvd_median_rel_err_top200=float(np.median(relerr))))
        print('B',name,rows[-1],round(time.time()-t0),'s',flush=True)
    out['B'][name]=rows; json.dump(out,open(f'{R}/robustness.json','w'),indent=1)
print('done')
