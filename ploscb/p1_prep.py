"""PLOS Computational Biology revision, step 1: node tables and connectivity matrices (synapse counts, pairs >= 5 synapses)
for the two further connectomes.
  BANC v888 (female brain + nerve cord; Bates et al. 2026): v3 neuron-to-neuron edgelist, whole CNS.
  MANC v1.0 (male nerve cord; Takemura et al. 2024; Marin et al. 2024): traced-connections adjacency, all traced neurons.
MANC neurons receive the MaleCNS nerve-cord type name (mancType) through the MaleCNS mancGroup field, which equals the MANC
group identifier, so that MaleCNS, MANC and BANC nerve-cord types share one namespace.
Writes ploscb/data/{banc,manc}_nodes.parquet and {banc_cns,manc_vnc}_w5.npz."""
import os, time, numpy as np, pandas as pd, pyarrow.feather as pf, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.environ.get('FLYBRAIN_DATA',os.path.join(os.path.dirname(HERE),'data','raw')); OUT=os.path.join(HERE,'data'); T=5
SIGN={'acetylcholine':1,'dopamine':1,'octopamine':1,'serotonin':1,'histamine':1,'tyramine':1,'gaba':-1,'glutamate':-1}
t0=time.time()
def csr(pre,post,w,ids,n):
    idmap=pd.Series(np.arange(n),index=ids); m=pd.Series(pre).isin(idmap.index).values&pd.Series(post).isin(idmap.index).values
    M=sp.csr_matrix((w[m].astype(np.float64),(idmap.reindex(pre[m]).values,idmap.reindex(post[m]).values)),shape=(n,n)); M.sum_duplicates(); return M
# ---------------- BANC
b=pd.read_feather(f'{D}/banc/banc_888_meta.feather')
b=b[b.super_class.notna()&~b.super_class.isin(['glia','not_a_neuron','trachea'])&b.region.isin(['central_brain','optic_lobe','ventral_nerve_cord'])
    &~b.status.astype(str).str.contains('DUPLICATED')].copy()
b['bodyId']=b.banc_888_id.astype(str); b=b.sort_values('bodyId').reset_index(drop=True); b['idx']=np.arange(len(b))
b['region3']=b.region.map({'central_brain':'central','optic_lobe':'optic','ventral_nerve_cord':'vnc'})
b['somaSide']=b.side.map({'left':'L','right':'R'}); b['sign']=b.neurotransmitter_predicted.map(SIGN).fillna(1).astype(int)
b['dimorphism']=b.sexually_dimorphic
nodes=b[['bodyId','idx','region3','super_class','cell_class','cell_type','fafb_cell_type','manc_cell_type','malecns_cell_type','somaSide','dimorphism','neurotransmitter_predicted','sign','hemilineage']].rename(columns={'region3':'region','super_class':'superclass','cell_class':'class','cell_type':'type','neurotransmitter_predicted':'nt'})
nodes.to_parquet(f'{OUT}/banc_nodes.parquet'); print('BANC nodes',len(nodes),nodes.region.value_counts().to_dict(),flush=True)
e=pf.read_table(f'{D}/banc/banc_888_edgelist_simple_v3.feather',columns=['pre','post','count']).to_pandas(); e=e[e['count']>=T]
M=csr(e.pre.values,e.post.values,e['count'].values,nodes.bodyId.values,len(nodes)); sp.save_npz(f'{OUT}/banc_cns_w{T}.npz',M)
print('BANC edges>=5',M.nnz,'synapses',int(M.sum()),round(time.time()-t0),'s',flush=True)
# ---------------- MANC
m=pd.read_feather(f'{D}/manc/manc-v1.0-neuron-properties.feather')
m=m[(m.status=='Traced')&(m['class'].astype(str)!='Glia')].copy(); m=m.sort_values('bodyId').reset_index(drop=True); m['idx']=np.arange(len(m))
m['somaSide']=m.somaSide.map({'LHS':'L','RHS':'R'}).fillna(m.rootSide.map({'LHS':'L','RHS':'R'}))
m['sign']=m.predictedNt.astype(str).str.lower().map(SIGN).fillna(1).astype(int)
a=pd.read_feather(f'{D}/body-annotations-male-cns-v1.0-minconf-0.5.feather',columns=['mancGroup','mancType'])
g2t=a.dropna().assign(g=lambda x: pd.to_numeric(x.mancGroup,errors='coerce')).dropna(subset=['g']).groupby('g').mancType.agg(lambda s: s.value_counts().idxmax())
m['mancType']=pd.to_numeric(m.group,errors='coerce').map(g2t)
m['region']='vnc'
mn=m[['bodyId','idx','region','class','subclass','type','group','mancType','somaSide','predictedNt','sign','hemilineage','birthtime']].rename(columns={'predictedNt':'nt'})
mn.to_parquet(f'{OUT}/manc_nodes.parquet'); print('MANC nodes',len(mn),'with MaleCNS type via group',int(mn.mancType.notna().sum()),flush=True)
c=pd.read_csv(f'{D}/manc/traced-connections.csv'); c=c[c.weight>=T]
M=csr(c.bodyId_pre.values,c.bodyId_post.values,c.weight.values,mn.bodyId.values,len(mn)); sp.save_npz(f'{OUT}/manc_vnc_w{T}.npz',M)
print('MANC edges>=5',M.nnz,'synapses',int(M.sum()),round(time.time()-t0),'s',flush=True)
