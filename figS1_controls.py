from figlib import *
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.gridspec as gridspec, json
R='results'; S=json.load(open(f'{R}/spectrum_summary.json')); MS=json.load(open(f'{R}/male_specific.json')); TZ=np.load(f'{R}/type_spectra.npz')
fig=plt.figure(figsize=(174*MM,60*MM)); gs=gridspec.GridSpec(1,4,wspace=0.6,left=0.075,right=0.985,top=0.86,bottom=0.28)
axA,axB,axC,axD=[fig.add_subplot(gs[0,i]) for i in range(4)]
# A: localisation of modes
z=np.load(f'{R}/mode_localisation.npz'); pu=z['pr_left']; pv=z['pr_right']   # derived from modes_male_cns_count.npz (see male_specific.py)
axA.plot(np.arange(1,len(pu)+1),pu,'.',ms=3,color=Q['male'],label='Output (left) vectors'); axA.plot(np.arange(1,len(pv)+1),pv,'.',ms=3,color=Q['male_light'],label='Input (right) vectors')
axA.set_yscale('log'); plain_log(axA,'y'); axA.set_ylim(1,3000); axA.set_xlabel('Mode rank'); axA.set_ylabel('Neurons per mode'); axA.legend(fontsize=5.5,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=1,borderaxespad=0.1,handletextpad=0.3)
# B: signed, input-normalised PR/N by region, with renormalised degree-preserving nulls where computed
regions=[('male_cns','CNS'),('male_brain','Brain'),('male_central','Central'),('male_optic','Optic'),('male_vnc','VNC'),('female_brain','Brain'),('female_central','Central'),('female_optic','Optic')]
x=np.arange(len(regions)); vals=[S[f'{k}|signed_norm']['PR_over_N'] for k,_ in regions]; cols=[Q['male']]*5+[Q['female']]*3
axB.bar(x,vals,color=cols,width=0.62,label='Connectome')
nulls=[S[f'{k}|signed_norm'].get('null_config',{}).get('PR_over_N',np.nan) if S[f'{k}|signed_norm'].get('null_config',{}).get('renormalised') else np.nan for k,_ in regions]
axB.scatter(x,nulls,marker='_',s=120,color='k',lw=1.2,label='Degree-preserving null\n(renormalised)',zorder=3); axB.axhline(0.5,color=Q['iid'],lw=0.6,ls=':')
axB.set_yscale('log'); plain_log(axB,'y'); axB.set_ylim(2e-3,1.0); axB.set_xticks(x); axB.set_xticklabels([l for _,l in regions],rotation=90,fontsize=6); axB.set_ylabel('Participation ratio / N')
axB.text(2,0.55,'Male',color=Q['male'],ha='center',va='bottom',fontsize=6); axB.text(6,0.55,'Female',color=Q['female'],ha='center',va='bottom',fontsize=6); axB.legend(fontsize=5.2,loc='upper left',bbox_to_anchor=(0,0.80),handlelength=1.4)

# C: containment medians against subspace dimension (solid: outputs; dashed: inputs)
B=MS['B_containment']; ks=[50,200,1000]
for kind,ls in [('out','-'),('in','--')]:
    for grp,col,lab in [('iso_heldout_median',Q['iso'],'Isomorphic (held out)'),('ms_median',Q['ms'],'Male-specific'),('dim_median',Q['dim'],'Dimorphic')]:
        axC.plot(ks,[B[f'{kind}_k{k}'][grp] for k in ks],'o'+ls,ms=3,lw=0.9,color=col,label=lab if kind=='out' else None)
axC.set_xscale('log'); axC.set_yscale('log'); plain_log(axC); axC.set_ylim(1e-4,40); axC.set_yticks([1e-4,1e-3,1e-2,1e-1,1]); axC.set_xlabel('Subspace dimension k'); axC.set_ylabel('Median fraction of connectivity\ninside isomorphic subspace'); axC.legend(fontsize=5.2,loc='upper left',handlelength=1.6,labelspacing=0.25,title='solid: outputs; dashed: inputs',title_fontsize=5.2)
# D: cell-type-level count spectra with nulls
for key,col,lab,ls in [('male_cns_types|count',Q['male'],'Male types','-'),('female_brain_types|count',Q['female'],'Female types','-'),('male_cns_types|count|null_config',Q['null'],'Male null','--'),('female_brain_types|count|null_config',Q['female_light'],'Female null','--')]:
    s=TZ[key]; axD.plot(np.arange(1,len(s)+1),s/s[0],color=col,lw=1.0,ls=ls,label=lab)
axD.set_xscale('log'); axD.set_yscale('log'); plain_log(axD); axD.set_ylim(1e-4,1.5); axD.set_xlabel('Mode rank'); axD.set_ylabel('Singular value (fraction of largest)'); axD.legend(fontsize=5.2,loc='lower left',handlelength=1.4,borderaxespad=0.2); axD.set_title('Cell types, synapse counts',fontsize=7)
panel_labels_tight(fig,[axA,axB,axC,axD],'ABCD')
base='figures/FigureS1'; require_matplotlib_panel_alignment(fig,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True)
fig.savefig(base+'.pdf'); fig.savefig(base+'.svg'); fig.savefig(base+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(base+'.png',dpi=300); print('saved',base)
