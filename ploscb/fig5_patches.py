"""Fig 5. Sex-specific neurons of both sexes are wired like ordinary neurons and read from the shared subspace.
A  effective rank of the edge set touching the male-specific (MaleCNS) or female-specific (BANC) neurons, against 50 random sets
   of isomorphic neurons matched on superclass and synapse-count decile
B  loss of effective rank on removing the sex-specific neurons, against 50 matched random removals
C, D  containment: AUC of the fraction of each neuron's output (C) or input (D) vector captured by the leading k modes of the
   isomorphic network, sex-specific against held-out isomorphic neurons (AUC > 0.5: more inside the shared subspace)"""
from figlib_plos import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
CASES=[('male_brain','MaleCNS brain\nmale-specific','MaleCNS'),('male_cns','MaleCNS CNS\nmale-specific','MaleCNS'),('female_brain','BANC brain\nfemale-specific','BANC'),('female_cns','BANC CNS\nfemale-specific','BANC')]
P={c:load(os.path.join(RES,f'patch_{c}.json')) for c,_,_ in CASES}
fig=plt.figure(figsize=(W,118*MM)); gs=gridspec.GridSpec(2,2,left=0.14,right=0.935,top=0.95,bottom=0.09,wspace=0.62,hspace=0.5)
ax=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
y=np.arange(len(CASES))[::-1]; rng=np.random.default_rng(0)
for a,key,obs,rnd,xlabel,pk in [(ax[0],'patch_rank','PR','rand','effective rank of the patch (PR)','p_lower'),(ax[1],'removal','dPR','rand_dPR','loss of effective rank on removal','p_greater')]:
    for yi,(c,lab,ds) in zip(y,CASES):
        d=P[c][key]; r=np.array(d[rnd]); a.scatter(r,yi+rng.uniform(-0.18,0.18,len(r)),s=6,color='#BDBDBD',lw=0,zorder=2)
        a.scatter([d[obs]],[yi],s=40,marker='D',color=DS[ds],zorder=3,lw=0)
        a.text(1.02,yi,f"p = {d[pk]:.2f}",transform=a.get_yaxis_transform(),fontsize=8,va='center',ha='left',color=GREY)
    a.set_yticks(y); a.set_yticklabels([c[1] for c in CASES]); a.tick_params(axis='y',length=0); a.set_ylim(-0.6,len(CASES)-0.4); a.set_xlabel(xlabel)
ax[1].axvline(0,color=GREY,lw=0.6,ls=':')
ax[0].legend(handles=[Line2D([],[],marker='D',ls='',ms=5,color='#4D4D4D',label='sex-specific neurons'),Line2D([],[],marker='o',ls='',ms=3,color='#BDBDBD',label='matched random sets')],loc='lower right',borderaxespad=0.1,handletextpad=0.2,labelspacing=0.2)
KS=[50,200,1000]; MK={50:'o',200:'s',1000:'^'}; off={50:0.18,200:0,1000:-0.18}
for a,kind,title in [(ax[2],'out','outputs'),(ax[3],'in','inputs')]:
    for yi,(c,lab,ds) in zip(y,CASES):
        for k in KS:
            v=P[c]['containment'][f'{kind}_k{k}']; sig=v['p']<0.01
            a.scatter([v['auc']],[yi+off[k]],marker=MK[k],s=22,color=DS[ds] if sig else 'white',edgecolor=DS[ds],lw=1.0,zorder=3)
    a.axvline(0.5,color=GREY,lw=0.6,ls='--'); a.set_xlim(0.4,0.75); a.set_yticks(y); a.set_yticklabels([c[1] for c in CASES]); a.tick_params(axis='y',length=0); a.set_ylim(-0.6,len(CASES)-0.4)
    a.set_xlabel(f'containment AUC, {title}')
ax[3].legend(handles=[Line2D([],[],marker=MK[k],ls='',ms=5,color='#4D4D4D',label=f'k = {k:,}') for k in KS]+[Line2D([],[],marker='o',ls='',ms=5,mfc='white',mec='#4D4D4D',label='p ≥ 0.01')],loc='lower right',borderaxespad=0.1,handletextpad=0.2,labelspacing=0.2)
label_panels(fig,ax,'ABCD'); gate(fig,'Fig5',ax,'ABCD'); export(fig,'Fig5')
