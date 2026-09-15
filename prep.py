"""Build neuron-level sparse connectivity matrices (synapse counts, pair threshold >=5) for the male CNS (MaleCNS v1.0)
and the female brain (FlyWire v783), with node tables (superclass, type, side, dimorphism, neurotransmitter sign)."""
import os, time, numpy as np, pandas as pd, pyarrow as pa, pyarrow.feather as pf, pyarrow.compute as pc, scipy.sparse as sp
D=os.environ.get('FLYBRAIN_DATA',os.path.join('data','raw')); OUT='data'; T=5
t0=time.time()
# ---------------- male nodes
a=pd.read_feather(f'{D}/body-annotations-male-cns-v1.0-minconf-0.5.feather')
a=a[(a.status=='Traced')&a.superclass.notna()&~a.superclass.astype(str).str.contains('glia',case=False)].copy()
nt=pf.read_table(f'{D}/body-neurotransmitters-male-cns-v1.0.feather',columns=['body','consensus_nt','celltype_predicted_nt','predicted_nt']).to_pandas()
nt['nt']=nt.consensus_nt.fillna(nt.celltype_predicted_nt).fillna(nt.predicted_nt)
a=a.merge(nt[['body','nt']],left_on='bodyId',right_on='body',how='left')
SIGN={'acetylcholine':1,'dopamine':1,'octopamine':1,'serotonin':1,'histamine':1,'gaba':-1,'glutamate':-1}
a['sign']=a.nt.map(SIGN).fillna(1).astype(int)
VNC_SC={'vnc_intrinsic','vnc_sensory','vnc_motor','vnc_efferent','vnc_tbc','vnc_sensory_tbc','vnc_endocrine'}
a['region']=np.where(a.superclass.isin(VNC_SC),'vnc',np.where(a.superclass.isin({'ol_intrinsic','ol_sensory','visual_projection','visual_centrifugal','visual_projection_tbc'}),'optic','central'))
a=a.sort_values('bodyId').reset_index(drop=True); a['idx']=np.arange(len(a))
node=a[['bodyId','idx','superclass','class','type','flywireType','somaSide','dimorphism','fruDsx','itoleeHl','trumanHl','birthtime','nt','sign','region']].copy()
node.to_parquet(f'{OUT}/male_nodes.parquet'); print('male nodes',len(node),node.region.value_counts().to_dict(),round(time.time()-t0),'s',flush=True)
idmap=pd.Series(node.idx.values,index=node.bodyId.values)
def to_csr(pre,post,w,n):
    m=pd.Series(pre).isin(idmap.index).values&pd.Series(post).isin(idmap.index).values
    i=idmap.reindex(pre[m]).values; j=idmap.reindex(post[m]).values
    M=sp.csr_matrix((w[m].astype(np.float64),(i,j)),shape=(n,n)); M.sum_duplicates(); return M
# ---------------- male whole-CNS matrix from the pair table
w=pf.read_table(f'{D}/connectome-weights-male-cns-v1.0-minconf-0.5.feather',memory_map=True)
w=w.filter(pc.greater_equal(w['weight'],T)).to_pandas()
M=to_csr(w.body_pre.values,w.body_post.values,w.weight.values,len(node)); sp.save_npz(f'{OUT}/male_cns_w{T}.npz',M)
print('male CNS edges>=5',M.nnz,'synapses',int(M.sum()),round(time.time()-t0),'s',flush=True)
del w
# ---------------- male brain-only and VNC-only matrices from synapse ROI
VNC_ROI=set(['ANm','LTct','IntTct','VNC-unspecified','CV-unspecified','CRN']+[f'{x}({s})' for x in ['LegNp(T1)','LegNp(T2)','LegNp(T3)','WTct(UTct-T2)','HTct(UTct-T3)','NTct(UTct-T1)','Ov','mVAC(T1)','mVAC(T2)','mVAC(T3)','AB','DMetaN','MesoLN','MetaLN','AbNT','ADMN','AbN1','AbN2','AbN3','AbN4','PDMN','ProLN','MesoAN','ProAN','VProN','CvN','DProN','ProCN','PrN'] for s in 'LR'])
t=pf.read_table(f'{D}/syn-partners-male-cns-v1.0-minconf-0.5.feather',columns=['body_pre','body_post','primary_post'],memory_map=True)
parts={'brain':[], 'vnc':[]}
for ch in t.to_batches(max_chunksize=20_000_000):
    roi=ch.column('primary_post'); dic=roi.dictionary.to_pylist(); codes=roi.indices.to_numpy(zero_copy_only=False)
    isv=np.array([d in VNC_ROI for d in dic]); isu=np.array([d=='<unspecified>' or d is None for d in dic])
    v=isv[codes]; b=~v&~isu[codes]
    pre=ch.column('body_pre').to_numpy(); post=ch.column('body_post').to_numpy()
    for k,m in (('brain',b),('vnc',v)):
        df=pd.DataFrame({'pre':pre[m],'post':post[m]}); parts[k].append(df.groupby(['pre','post'],sort=False).size().reset_index(name='w'))
    print('chunk',len(pre),round(time.time()-t0),'s',flush=True)
for k in parts:
    df=pd.concat(parts[k]).groupby(['pre','post'],sort=False).w.sum().reset_index(); df=df[df.w>=T]
    M=to_csr(df.pre.values,df.post.values,df.w.values,len(node)); sp.save_npz(f'{OUT}/male_{k}_w{T}.npz',M); print(k,'edges>=5',M.nnz,'synapses',int(M.sum()),flush=True)
del t,parts
# ---------------- female (FlyWire v783)
c=pd.read_csv(f'{D}/flywire/classification.csv.gz'); ct=pd.read_csv(f'{D}/flywire/consolidated_cell_types.csv.gz'); nn=pd.read_csv(f'{D}/flywire/neurons.csv.gz',usecols=['root_id','nt_type'])
f=c.merge(ct[['root_id','primary_type']],on='root_id',how='left').merge(nn,on='root_id',how='left')
FS={'ACH':1,'DA':1,'SER':1,'OCT':1,'GABA':-1,'GLUT':-1}; f['sign']=f.nt_type.map(FS).fillna(1).astype(int)
f['region']=np.where(f.super_class.isin(['optic','visual_projection','visual_centrifugal']),'optic','central')
f=f.sort_values('root_id').reset_index(drop=True); f['idx']=np.arange(len(f)); f['somaSide']=f.side.map({'left':'L','right':'R','center':'M'})
f.rename(columns={'root_id':'bodyId','super_class':'superclass','primary_type':'type'})[['bodyId','idx','superclass','class','sub_class','hemilineage','somaSide','type','nt_type','sign','region']].to_parquet(f'{OUT}/female_nodes.parquet')
e=pd.read_csv(f'{D}/flywire/connections.csv.gz',usecols=['pre_root_id','post_root_id','syn_count']).groupby(['pre_root_id','post_root_id'],sort=False).syn_count.sum().reset_index(); e=e[e.syn_count>=T]
fid=pd.Series(f.idx.values,index=f.bodyId.values if 'bodyId' in f else f.root_id.values)
m=e.pre_root_id.isin(fid.index)&e.post_root_id.isin(fid.index); e=e[m]
M=sp.csr_matrix((e.syn_count.values.astype(np.float64),(fid.reindex(e.pre_root_id).values,fid.reindex(e.post_root_id).values)),shape=(len(f),len(f))); M.sum_duplicates(); sp.save_npz(f'{OUT}/female_brain_w{T}.npz',M)
print('female nodes',len(f),'edges>=5',M.nnz,'synapses',int(M.sum()),round(time.time()-t0),'s',flush=True)
