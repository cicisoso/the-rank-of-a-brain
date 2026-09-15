from figlib import *
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.gridspec as gridspec, json
R='results'; MS=json.load(open(f'{R}/male_specific.json')); TS=json.load(open(f'{R}/types_sex.json'))
fig=plt.figure(figsize=(174*MM,60*MM)); gs=gridspec.GridSpec(1,4,wspace=0.6,left=0.07,right=0.99,top=0.86,bottom=0.26)
axA,axD,axC,axB=[fig.add_subplot(gs[0,i]) for i in range(4)]
# A: schematic of the question: isomorphic matrix plus a sex-specific border ('patch'); numbers from the analysis
c=MS['C_touching']; a=MS['A_remove_ms']
axA.set_xlim(0,10); axA.set_ylim(-1.5,10); axA.axis('off')
from matplotlib.patches import Rectangle
axA.add_patch(Rectangle((1,2),6,6,facecolor='#E6E6E6',edgecolor='k',lw=0.6)); axA.add_patch(Rectangle((7,2),1.2,6,facecolor=Q['ms'],edgecolor='k',lw=0.6,alpha=0.9)); axA.add_patch(Rectangle((1,0.8),7.2,1.2,facecolor=Q['ms'],edgecolor='k',lw=0.6,alpha=0.9))
axA.text(4,5,'Isomorphic\nneurons',ha='center',va='center',fontsize=6.5); axA.text(7.6,8.35,'Male-\nspecific',ha='center',va='bottom',fontsize=6,color=Q['ms'])
axA.text(4.6,-0.25,f"Patch: {c['n_ms']:,} neurons, {100*MS['syn_ms']/MS['syn_total']:.1f}% of synapses",ha='center',va='top',fontsize=6)
axA.set_title('Does the sex-specific patch add rank?',fontsize=7)
# B: rank change on removal: male-specific vs matched random
b=MS['A_remove_ms_dim']
x=np.arange(2); axB.bar(x-0.18,[a['dPR'],b['dPR']],width=0.34,color=Q['ms'],label='Sex-specific neurons'); axB.bar(x+0.18,[a['rand_dPR_mean'],b['rand_dPR_mean']],yerr=[a['rand_dPR_sd'],b['rand_dPR_sd']],width=0.34,color=Q['iso'],label='Matched random neurons',capsize=2,error_kw=dict(lw=0.6))
axB.axhline(0,color='k',lw=0.5); axB.set_ylim(-70,230); axB.set_xticks(x); axB.set_xticklabels([f"Male-specific\n(n = {a['n_removed']:,})",f"+ dimorphic\n(n = {b['n_removed']:,})"],fontsize=6); axB.set_ylabel('Loss of participation ratio\non removal'); axB.legend(fontsize=5.5,loc='upper right'); axB.set_title('Removing the patch',fontsize=7)
# C: containment in the isomorphic subspace (k = 200), output and input vectors
co=np.load(f'{R}/containment_out.npz'); ci=np.load(f'{R}/containment_in.npz')
data=[co['iso'],co['ms'],co['dim'],ci['iso'],ci['ms'],ci['dim']]; pos=[0,1,2,3.4,4.4,5.4]; cols=[Q['iso'],Q['ms'],Q['dim']]*2
bp=axC.boxplot(data,positions=pos,widths=0.7,whis=(5,95),showfliers=False,patch_artist=True,medianprops=dict(color='k',lw=0.8),whiskerprops=dict(lw=0.5),capprops=dict(lw=0.5))
for p,c_ in zip(bp['boxes'],cols): p.set_facecolor(c_); p.set_edgecolor('k'); p.set_linewidth(0.5)
axC.set_yscale('log'); plain_log(axC,'y'); axC.set_xticks([1,4.4]); axC.set_xticklabels(['Outputs','Inputs']); axC.set_ylabel('Fraction of connectivity inside\nisomorphic 200-mode subspace')
from matplotlib.patches import Patch
axC.legend(handles=[Patch(color=Q['iso'],label='Isomorphic (held out)'),Patch(color=Q['ms'],label='Male-specific'),Patch(color=Q['dim'],label='Dimorphic')],fontsize=5.2,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3,columnspacing=0.6,handletextpad=0.4)
B=MS['B_containment']; axC.set_title(f"AUC out {B['out_k200']['auc_ms_vs_iso']:.2f}, in {B['in_k200']['auc_ms_vs_iso']:.2f}",fontsize=7)
# D: effective rank of the edge set touching male-specific neurons vs matched random sets
axD.bar([0,1],[c['PR_ms'],c['rand_PR_mean']],yerr=[0,c['rand_PR_sd']],color=[Q['ms'],Q['iso']],width=0.6,capsize=2,error_kw=dict(lw=0.6))
axD.set_xticks([0,1]); axD.set_xticklabels(['Male-specific\npatch','Matched\nrandom'],fontsize=6); axD.set_ylabel('Participation ratio of edge set'); axD.set_title('Rank of the patch',fontsize=7)
panel_labels_tight(fig,[axA,axD,axC,axB],'ABCD')
base='figures/Figure4'; require_matplotlib_panel_alignment(fig,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,axes=[axA,axD,axC,axB],panel_ids=['A','B','C','D'],exemptions=[{'panels':['A'],'checks':['panel-width'],'reason':'schematic hero panel'}])
fig.savefig(base+'.pdf'); fig.savefig(base+'.svg'); fig.savefig(base+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(base+'.png',dpi=300); print('saved',base)
