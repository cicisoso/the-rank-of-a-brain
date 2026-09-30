"""S1 Table (Excel workbook with every number behind the figures and text) and Table 1 of the manuscript (markdown).
Output: tables/S1_Table.xlsx and results/table1.md"""
import os, json, numpy as np, pandas as pd, scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); R=os.path.join(HERE,'results'); QR=os.path.join(Q,'results')
OUT=os.path.join(HERE,'tables'); os.makedirs(OUT,exist_ok=True)
L=lambda p: json.load(open(p))
PT=pd.DataFrame(L(f'{R}/pr_table.json')); BP=L(f'{R}/block_precise.json'); RB=L(f'{R}/robustness.json'); SM=L(f'{R}/scaling_modes.json')
SI=L(f'{R}/sex_individual.json'); SIM=L(f'{R}/similarity.json'); SMODE=L(f'{R}/sex_mode.json'); TB=L(f'{R}/types_banc_manc.json')
SQ=L(f'{QR}/spectrum_summary.json'); SG=L(f'{QR}/signal_rank.json')
spec={}
for f in ['banc_cns','banc_brain','banc_central','banc_optic','banc_vnc','manc_all','manc_vnc']: spec.update(L(f'{R}/spectrum_{f}.json'))
# ---- spectral summaries (leading singular values, stable rank, energy ranks, modes above the null edge)
rows=[]
for k,v in SQ.items():
    net,flav=k.split('|'); ds='MaleCNS' if net.startswith('male') else 'FlyWire'
    rows.append(dict(dataset=ds,network=net,flavour=flav,N=v['N'],nnz=v['nnz'],stable_rank=v['stable_rank'],modes_50pct_energy=v['rank_50'],modes_80pct_energy=v['rank_80'],energy_in_leading_k=v['energy_topk'],k=v['k'],
                     modes_above_config_edge=SG.get(k,{}).get('null_config',{}).get('n_above'),modes_above_wshuf_edge=SG.get(k,{}).get('null_wshuf',{}).get('n_above'),
                     PR_48probe=v['PR'],PR_null_config=v.get('null_config',{}).get('PR'),PR_null_wshuf=v.get('null_wshuf',{}).get('PR'),PR_null_random_placement=v.get('null_er',{}).get('PR')))
for k,v in spec.items():
    net,flav=k.split('|'); ds='BANC' if net.startswith('banc') else 'MANC'
    rows.append(dict(dataset=ds,network=net,flavour=flav,N=v['N'],nnz=v['nnz'],stable_rank=v['stable_rank'],modes_50pct_energy=v['rank_50'],modes_80pct_energy=v['rank_80'],energy_in_leading_k=v['energy_topk'],k=v['k'],
                     modes_above_config_edge=v['null_config']['n_above_edge'],modes_above_wshuf_edge=v['null_wshuf']['n_above_edge'],
                     PR_48probe=v['PR'],PR_null_config=v['null_config']['PR'],PR_null_wshuf=v['null_wshuf']['PR'],PR_null_random_placement=v['null_er']['PR']))
spectra=pd.DataFrame(rows)
# ---- Table 1
whole=[('MaleCNS','CNS','male_cns','MaleCNS v1.0','male','brain + nerve cord'),('FlyWire','brain','female_brain','FlyWire v783','female','brain'),
       ('BANC','CNS','banc_cns','BANC v888','female','brain + nerve cord'),('MANC','nerve cord (all traced)','manc_all','MANC v1.0','male','nerve cord')]
t1=[]
for ds,reg,net,rel,sex,cov in whole:
    p=PT[(PT.dataset==ds)&(PT.region==reg)&(PT.flavour=='count')].iloc[0]; s=spectra[(spectra.network==net)&(spectra.flavour=='count')].iloc[0]; b=BP[ds]
    t1.append({'Connectome':rel,'Sex':sex,'Coverage':cov,'Neurons':f"{int(p.N):,}",'Connections (≥5 synapses)':f"{int(p.nnz):,}",'Effective rank, PR (± s.e.)':f"{p.PR:,.0f} ± {p.PR_se:.0f}",
               'PR / N (%)':f"{100*p.PR_over_N:.2f}",'Degree-preserving null PR':f"{p.PR_null:,.0f}",'Null / connectome':f"{p.null_over_real:.1f}",'Stable rank':f"{s.stable_rank:.0f}",
               'Modes above null edge':f"{int(s.modes_above_config_edge)}",'Energy explained by types × side (%)':f"{100*b['type_side']['frac']:.0f}"})
T1=pd.DataFrame(t1)
cols=list(T1.columns); md='| '+' | '.join(cols)+' |\n|'+'---|'*len(cols)+'\n'+''.join('| '+' | '.join(str(r[c]) for c in cols)+' |\n' for _,r in T1.iterrows())
open(f'{R}/table1.md','w').write(md); print(md)
# ---- remaining sheets
rob=pd.DataFrame([dict(dataset=d,weights=t,**v) for d,x in RB['A'].items() for t,v in x.items()])
val=pd.DataFrame([dict(network=k,**r) for k,rows in RB['B'].items() for r in rows])
scal=pd.DataFrame([dict(dataset=d,**r,slope=x['slope']) for d,x in SM['scaling'].items() for r in x['rows']])
modes=pd.DataFrame([dict(dataset=d,median_neurons_out_top200=m['neurons_out_median_top200'],median_neurons_in_top200=m['neurons_in_median_top200'],median_neurons_out_top20=m['neurons_out_median_top20'],
                         median_neurons_in_top20=m['neurons_in_median_top20'],frac_leading100_modes_single_region=m['frac_modes_single_region']) for d,m in SM['modes'].items()])
blk=pd.DataFrame([dict(dataset=d,grouping=g,PR_full=v['PR_full'],PR_full_se=v['PR_full_se'],n_typed=v['n_typed'],n_types=v['n_types'],**v[g]) for d,v in BP.items() for g in ['type','type_side','random']])
tm=pd.DataFrame([dict(dataset=k.split('_')[0],types=v['N'],PR=v['PR'],PR_over_N=v['PR_over_N'],stable_rank=v['stable_rank'],modes_90pct=v['rank_90'],null_PR=v['null_config']['PR'],modes_above_null_edge=v['n_above_edge']) for k,v in TB.items() if k.endswith('_types|signed_norm')])
cap=pd.DataFrame([dict(region=reg,pair=p,kind=v['kind'],**{f'k={k}':x for k,x in zip(SI[reg]['k'],v['rel'])}) for reg in SI for p,v in SI[reg]['capture'].items()])
sim=pd.DataFrame([dict(region=reg,**r) for reg in SIM for r in SIM[reg]['rows']])
diff=pd.DataFrame([dict(region=reg,difference=k,**v) for reg in SI for k,v in SI[reg]['diff'].items()])
mode=pd.DataFrame([dict(comparison=k,sigma_1=v['sigma'][0],sigma_2=v['sigma'][1],top_types=', '.join(v['top_types']),fru_dsx_in_top1pct=f"{v['fru_top1pct'][0]}/{v['fru_top1pct'][1]}",fru_p=v['fru_p'],dim_p=v['dim_p'],
                       superclass_top1pct=json.dumps(v['superclass_of_top1pct'])) for k,v in SMODE.items() if k.startswith('MaleCNS')])
agree=pd.DataFrame([dict(hemisphere=k.split()[-1],**v) for k,v in SMODE.items() if k.startswith('agreement')])
pat=[]
for c in ['male_brain','male_cns','female_brain','female_cns']:
    d=L(f'{R}/patch_{c}.json'); base={k:d[k] for k in ['case','N','n_patch','n_dim','n_iso','syn_frac_patch','n_draws']}
    pat.append(dict(**base,patch_PR=d['patch_rank']['PR'],random_PR_mean=d['patch_rank']['rand_mean'],random_PR_sd=d['patch_rank']['rand_sd'],p_patch_lower=d['patch_rank']['p_lower'],
                    removal_dPR=d['removal']['dPR'],random_dPR_mean=d['removal']['rand_dPR_mean'],random_dPR_sd=d['removal']['rand_dPR_sd'],p_removal_greater=d['removal']['p_greater'],
                    **{f'AUC_{k}':v['auc'] for k,v in d['containment'].items()},**{f'p_{k}':v['p'] for k,v in d['containment'].items()}))
pat=pd.DataFrame(pat)
readme=['S1 Table. Numerical results behind every figure and statement of "The effective rank of complete fly connectomes is set by cell types and shared between the sexes".',
        'Connectomes: MaleCNS v1.0 (male CNS), FlyWire v783 (female brain), BANC v888 (female brain and nerve cord; v3 edgelist), MANC v1.0 (male nerve cord). Connections with at least 5 synapses; weights are synapse counts unless stated.',
        'PR = participation ratio (sum sigma^2)^2 / sum sigma^4; sum sigma^4 estimated with 1,024 Rademacher probes (Hutchinson); s.e. from the probe variance. Null = degree-preserving rewiring (renormalised after rewiring for signed, input-normalised weights).',
        'Sheets: Table1; effective_rank (all networks, both weightings, 1,024-probe estimates); spectra (stable rank, modes for 50/80% of energy, modes above the null edge, and 48/32-probe effective ranks of the connectome and of the degree-preserving, weight-shuffled and random-placement nulls); robustness (thresholds and weights); estimator_validation; scaling; mode_localisation; block_model; type_matrices; capture (relative capture of shared-type matrices, all pairs and k); similarity (entrywise, reliability-corrected); difference_spectra; sex_mode (leading male-female difference mode); sex_mode_agreement; sex_specific_patches.']
with pd.ExcelWriter(f'{OUT}/S1_Table.xlsx') as w:
    pd.DataFrame({'note':readme}).to_excel(w,sheet_name='README',index=False)
    for name,df in [('Table1',T1),('effective_rank',PT),('spectra',spectra),('robustness',rob),('estimator_validation',val),('scaling',scal),('mode_localisation',modes),('block_model',blk),
                    ('type_matrices',tm),('capture',cap),('similarity',sim),('difference_spectra',diff),('sex_mode',mode),('sex_mode_agreement',agree),('sex_specific_patches',pat)]:
        df.to_excel(w,sheet_name=name,index=False)
print('wrote S1_Table.xlsx')
