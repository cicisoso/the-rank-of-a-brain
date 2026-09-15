from figlib import *
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.gridspec as gridspec, json
R='results'; TS=json.load(open(f'{R}/types_sex.json')); AL=np.load(f'{R}/alignment.npz'); TZ=np.load(f'{R}/type_spectra.npz'); SF=json.load(open(f'{R}/sex_followup.json'))
fig=plt.figure(figsize=(174*MM,60*MM)); gs=gridspec.GridSpec(1,4,wspace=0.55,left=0.085,right=0.99,top=0.86,bottom=0.26)
axA,axB,axC,axD=[fig.add_subplot(gs[0,i]) for i in range(4)]
# A: shared-type spectra male vs female (input fractions)
Sm=AL['Sm']; Sf=AL['Sf']
axA.plot(np.arange(1,len(Sm)+1),Sm/Sm[0],color=Q['male'],lw=1.1,label=f"Male (PR = {TS['shared_male']['PR']:.0f})"); axA.plot(np.arange(1,len(Sf)+1),Sf/Sf[0],color=Q['female'],lw=1.1,label=f"Female (PR = {TS['shared_female']['PR']:.0f})")
axA.set_xscale('log'); axA.set_yscale('log'); plain_log(axA); axA.set_ylim(1e-4,2); axA.set_xlabel('Mode rank'); axA.set_ylabel('Singular value (fraction of largest)'); axA.legend(fontsize=5.5,loc='lower left'); axA.set_title(f"{TS['shared_male']['N']:,} shared cell types",fontsize=7)
# B: alignment heatmap of the leading 50 left singular vectors
O=AL['O_left'][:50,:50]; im=axB.imshow(O,cmap='Blues',vmin=0,vmax=1,origin='upper',aspect='auto'); axB.set_xlabel('Female mode'); axB.set_ylabel('Male mode'); axB.set_title('Alignment of leading modes',fontsize=7)
cax=axB.inset_axes([0.55,0.86,0.4,0.05]); cb=fig.colorbar(im,cax=cax,orientation='horizontal'); cb.ax.tick_params(labelsize=5,length=1.5,pad=1); cb.set_label('Overlap',fontsize=5.5,labelpad=1)
# C: relative cross-projection (captured by the other matrix's leading modes / captured by own modes), hemisphere level
k=SF['k']
axC.plot(k,SF['male_L_by_male_R_rel'],'o-',ms=3,lw=0.9,color=Q['male'],label='M left by M right')
axC.plot(k,SF['female_L_by_female_R_rel'],'o-',ms=3,lw=0.9,color=Q['female'],label='F left by F right')
axC.plot(k,SF['male_L_by_female_L_rel'],'s--',ms=3,lw=0.9,color=Q['ms'],label='M left by F left')
axC.plot(k,SF['male_R_by_female_R_rel'],'s--',ms=3,lw=0.9,color=Q['dim'],label='M right by F right')
axC.set_xscale('log'); plain_log(axC,'x'); axC.set_xlabel('Subspace dimension k'); axC.set_ylabel('Energy captured by other\n(relative to own modes)'); axC.set_ylim(0,1.02); axC.legend(fontsize=5.2,loc='lower left',handlelength=1.4,ncol=1,labelspacing=0.25,borderaxespad=0.2); axC.set_title('Shared subspaces',fontsize=7)
# D: spectrum of the between-animal difference vs within-animal left-right differences (hemisphere-level, like for like)
for key,col,lab,ls in [('diff_male_L_vs_female_L',Q['ms'],'Male left − female left','-'),('diff_male_LR',Q['male'],'Male left − right','--'),('diff_female_LR',Q['female'],'Female left − right','--')]:
    s_=TZ[key]; axD.plot(np.arange(1,len(s_)+1),s_,color=col,lw=1.0,ls=ls,label=lab)
edge=max(TZ['diff_male_LR'][0],TZ['diff_female_LR'][0]); n_above=int((TZ['diff_male_L_vs_female_L']>edge).sum()); axD.axhline(edge,color='k',lw=0.5,ls=':')
axD.set_xscale('log'); axD.set_yscale('log'); plain_log(axD); axD.set_ylim(1e-2,10); axD.set_xlabel('Mode rank'); axD.set_ylabel('Singular value of difference'); axD.legend(fontsize=5.2,loc='lower left',handlelength=1.8); axD.set_title(f'{n_above} sex modes above noise',fontsize=7)
panel_labels_tight(fig,[axA,axB,axC,axD],'ABCD')
base='figures/Figure3'; require_matplotlib_panel_alignment(fig,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,axes=[axA,axB,axC,axD],panel_ids=['A','B','C','D'],exclude_axes=[cax])
fig.savefig(base+'.pdf'); fig.savefig(base+'.svg'); fig.savefig(base+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(base+'.png',dpi=300); print('saved',base)
