from figlib import *
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
matplotlib.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
import matplotlib.gridspec as gridspec, json
from speclib import quarter_circle_energy
R='results'; S=json.load(open(f'{R}/spectrum_summary.json')); Z=np.load(f'{R}/spectra.npz')
fig=plt.figure(figsize=(174*MM,62*MM)); gs=gridspec.GridSpec(1,4,wspace=0.55,left=0.06,right=0.99,top=0.86,bottom=0.24)
axA,axB,axC,axD=[fig.add_subplot(gs[0,i]) for i in range(4)]
# A: singular value spectrum of the male CNS (counts) vs null models
s=Z['male_cns|count']; r=np.arange(1,len(s)+1); N=S['male_cns|count']['N']
axA.plot(r,s/s[0],color=Q['male'],lw=1.2,label='Male CNS')
for nm,col,lab in [('null_config',Q['null'],'Degree-preserving'),('null_wshuf',Q['null_light'],'Weight-shuffled'),('null_er',Q['iid'],'Erdős–Rényi')]:
    k=f'male_cns|count|{nm}'
    if k in Z: sn=Z[k]; axA.plot(np.arange(1,len(sn)+1),sn/s[0],color=col,lw=0.9,ls='--' if nm=='null_er' else '-',label=lab)
axA.set_xscale('log'); axA.set_yscale('log'); plain_log(axA); axA.set_ylim(0.05,4); axA.set_xlabel('Mode rank'); axA.set_ylabel('Singular value (fraction of largest)'); axA.legend(fontsize=5.5,loc='upper right',handlelength=1.6)
axA.set_title('Whole male CNS, synapse counts',fontsize=7)
# B: cumulative energy vs rank, male CNS vs female brain vs nulls, with i.i.d. quarter circle
def curve(key):
    s=Z[key]; f2=S[key.split('|null')[0]]['frob2'] if 'null' in key else S[key]['frob2']
    return np.arange(1,len(s)+1), np.cumsum(s**2)/f2
for key,col,lab,ls in [('male_cns|count',Q['male'],'Male CNS','-'),('female_brain|count',Q['female'],'Female brain','-'),('male_cns|count|null_config',Q['null'],'Male, degree-preserving','--'),('female_brain|count|null_config',Q['female_light'],'Female, degree-preserving','--')]:
    if key in Z: x,c=curve(key); axB.plot(x,c,color=col,lw=1.1,ls=ls,label=lab)
axB.plot(np.arange(1,3001),quarter_circle_energy(N,3000),color=Q['iid'],lw=0.8,ls=':',label='i.i.d. (quarter circle)')
axB.set_xscale('log'); plain_log(axB,'x'); axB.set_xlabel('Number of leading modes'); axB.set_ylabel('Fraction of energy captured'); axB.set_ylim(0,1)
axB.legend(fontsize=5.2,loc='upper left',handlelength=1.6); axB.set_title('Energy captured by leading modes',fontsize=7)
# C: PR/N by region and sex, with degree-preserving null
regions=[('male_cns','CNS'),('male_brain','Brain'),('male_central','Central'),('male_optic','Optic'),('male_vnc','VNC'),('female_brain','Brain'),('female_central','Central'),('female_optic','Optic')]
x=np.arange(len(regions)); vals=[S[f'{k}|count']['PR_over_N'] for k,_ in regions]; nulls=[S[f'{k}|count'].get('null_config',{}).get('PR_over_N',np.nan) for k,_ in regions]
cols=[Q['male']]*5+[Q['female']]*3
axC.bar(x,vals,color=cols,width=0.62,label='Connectome'); axC.scatter(x,nulls,marker='_',s=120,color='k',lw=1.2,label='Degree-preserving null',zorder=3); axC.axhline(0.5,color=Q['iid'],lw=0.6,ls=':')
axC.text(len(regions)-0.5,0.55,'i.i.d.',fontsize=5.5,ha='right',va='bottom',color=Q['iid'])
axC.set_yscale('log'); plain_log(axC,'y'); axC.set_ylim(2e-3,1.0)
axC.set_xticks(x); axC.set_xticklabels([l for _,l in regions],rotation=90,fontsize=6); axC.set_ylabel('Participation ratio / N')
axC.text(2,1.1,'Male',color=Q['male'],ha='center',va='bottom',fontsize=6); axC.text(6,1.1,'Female',color=Q['female'],ha='center',va='bottom',fontsize=6); axC.legend(fontsize=5.2,loc='upper left',bbox_to_anchor=(0,0.80),handlelength=1.4)
# D: signed, input-normalised flavour: male CNS vs female brain spectra normalised by sigma_1 with nulls
for key,col,lab in [('male_cns|signed_norm',Q['male'],'Male CNS'),('female_brain|signed_norm',Q['female'],'Female brain'),('male_cns|signed_norm|null_config',Q['null'],'Male null'),('female_brain|signed_norm|null_config',Q['female_light'],'Female null')]:
    if key in Z: s_=Z[key]; axD.plot(np.arange(1,len(s_)+1),s_/s_[0],color=col,lw=1.0,ls='--' if 'null' in key else '-',label=lab)
axD.set_xscale('log'); axD.set_yscale('log'); plain_log(axD); axD.set_ylim(0.03,4); axD.set_xlabel('Mode rank'); axD.set_ylabel('Singular value (fraction of largest)'); axD.set_title('Signed, normalised weights',fontsize=7); axD.legend(fontsize=5.5,loc='upper right',handlelength=1.6)
panel_labels_tight(fig,[axA,axB,axC,axD],'ABCD')
base='figures/Figure1'; require_matplotlib_panel_alignment(fig,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True)
fig.savefig(base+'.pdf'); fig.savefig(base+'.svg'); fig.savefig(base+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(base+'.png',dpi=300); print('saved',base)
