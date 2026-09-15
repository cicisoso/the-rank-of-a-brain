from figlib import *
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.gridspec as gridspec, json
R='results'; S=json.load(open(f'{R}/spectrum_summary.json')); MS=json.load(open(f'{R}/male_specific.json')); TS=json.load(open(f'{R}/types_sex.json')); TZ=np.load(f'{R}/type_spectra.npz'); TR=json.load(open(f'{R}/type_redundancy.json'))
fig=plt.figure(figsize=(174*MM,60*MM)); gs=gridspec.GridSpec(1,4,width_ratios=[1.35,1,1,1],wspace=0.55,left=0.065,right=0.99,top=0.86,bottom=0.26)
axA,axB,axC,axD=[fig.add_subplot(gs[0,i]) for i in range(4)]
# A: region composition of the leading 60 modes of the whole CNS (stacked bars)
comp=MS['E_mode_composition']['region']; cats=comp['categories']; M=np.array(comp['matrix'])[:60]; order=[cats.index(c) for c in ['optic','central','vnc'] if c in cats]
bottom=np.zeros(len(M)); lab={'optic':'Optic lobes','central':'Central brain','vnc':'Nerve cord'}
for j in order: axA.bar(np.arange(1,len(M)+1),M[:,j],bottom=bottom,color=Q[cats[j]],width=1.0,lw=0,label=lab[cats[j]]); bottom+=M[:,j]
axA.set_xlim(0.5,len(M)+0.5); axA.set_ylim(0,1); axA.set_xlabel('Mode rank'); axA.set_ylabel('Share of mode energy'); axA.legend(fontsize=5.5,loc='upper center',bbox_to_anchor=(0.5,-0.3),ncol=3,columnspacing=0.8,handletextpad=0.4)
axA.set_title('Where the leading modes live',fontsize=7)
# B: rank scaling with network size
sc=MS['D_scaling']; n=np.array([d['N'] for d in sc]); pr=np.array([d['PR_mean'] for d in sc]); sd=np.array([d['PR_sd'] for d in sc])
axB.errorbar(n,pr,yerr=sd,fmt='o-',ms=3,lw=0.9,color=Q['male'],capsize=1.5,elinewidth=0.6,label='Random subnetworks')
axB.plot(n,pr[-1]*n/n[-1],ls='--',color=Q['null'],lw=0.8,label='Proportional to N'); axB.set_xscale('log'); axB.set_yscale('log'); plain_log(axB); axB.set_xlabel('Neurons in subnetwork'); axB.set_ylabel('Participation ratio'); axB.legend(fontsize=5.5,loc='lower right',handlelength=1.6); axB.set_title('Rank grows with size',fontsize=7)
# C: block model: energy explained by cell types (count matrices), male and female, plus participation ratios
labs=['Types','Types\n× side','Random']; xs=np.arange(3)
for j,(sex,col) in enumerate([('male',Q['male']),('female',Q['female'])]):
    fr=[TR[f'{sex}|count|type']['groups']['frac'],TR[f'{sex}|count|type_side']['groups']['frac'],TR[f'{sex}|count|type_side']['random_groups']['frac']]
    axC.bar(xs+(-0.19 if j==0 else 0.19),fr,width=0.36,color=col,label=(f"Male (PR {TR['male|count|type_side']['groups']['PR_block']:.0f} of {TR['male|count|type_side']['PR_full']:.0f})" if j==0 else f"Female (PR {TR['female|count|type_side']['groups']['PR_block']:.0f} of {TR['female|count|type_side']['PR_full']:.0f})"))
axC.set_xticks(xs); axC.set_xticklabels(labs,fontsize=6); axC.set_ylabel('Fraction of energy explained\nby block model'); axC.set_ylim(0,1); axC.legend(fontsize=5.2,loc='upper right',handlelength=1.2)
axC.set_title('Cell types explain the matrix',fontsize=7)
# D: type-level full spectra (male CNS types, female brain types) signed-normalised with config nulls
for key,col,lab,ls in [('male_cns_types|signed_norm',Q['male'],'Male types','-'),('female_brain_types|signed_norm',Q['female'],'Female types','-'),('male_cns_types|signed_norm|null_config',Q['null'],'Male null','--'),('female_brain_types|signed_norm|null_config',Q['female_light'],'Female null','--')]:
    if key in TZ: s=TZ[key]; axD.plot(np.arange(1,len(s)+1),s/s[0],color=col,lw=1.0,ls=ls,label=lab)
axD.set_xscale('log'); axD.set_yscale('log'); plain_log(axD); axD.set_ylim(1e-3,1.5); axD.set_xlabel('Mode rank'); axD.set_ylabel('Singular value (fraction of largest)'); axD.legend(fontsize=5.2,loc='lower left',handlelength=1.4,borderaxespad=0.2); axD.set_title('Cell-type level',fontsize=7)
panel_labels_tight(fig,[axA,axB,axC,axD],'ABCD')
base='figures/Figure2'; require_matplotlib_panel_alignment(fig,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,axes=[axA,axB,axC,axD],panel_ids=['A','B','C','D'],exemptions=[{'panels':['A'],'checks':['panel-width'],'reason':'wider stacked-bar hero panel'}])
fig.savefig(base+'.pdf'); fig.savefig(base+'.svg'); fig.savefig(base+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(base+'.png',dpi=300); print('saved',base)
