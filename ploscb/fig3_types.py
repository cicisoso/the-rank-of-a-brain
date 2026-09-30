"""Fig 3. Cell types set the rank, in all four connectomes.
A  block model: fraction of the energy of the count matrix explained by cell types, by cell types split by side, and by random
   groups of the same sizes
B  effective rank of the side-resolved block model against that of the full matrix
C  full singular value spectra of the signed, input-normalised cell-type matrices and of their degree-preserving nulls
D  effective rank of the cell-type matrices against their nulls"""
from figlib_plos import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
ORDER=['MaleCNS','FlyWire','BANC','MANC']
BP=load(os.path.join(RES,'block_precise.json')); TB=load(os.path.join(RES,'types_banc_manc.json'))
TS=np.load(os.path.join(RES,'type_spectra_banc_manc.npz'))
def blk(ds,grouping): return BP[ds][grouping]
TKEY={'MaleCNS':'malecns','FlyWire':'flywire','BANC':'banc','MANC':'manc'}
fig=plt.figure(figsize=(W,118*MM)); gs=gridspec.GridSpec(2,2,left=0.1,right=0.985,top=0.955,bottom=0.09,wspace=0.3,hspace=0.48)
ax=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
# A
a=ax[0]; x=np.arange(len(ORDER)); wd=0.26
for i,(g,lab,col) in enumerate([('type','cell types','#8C8C8C'),('type_side','cell types by side','#1A1A1A'),('random','random groups (side-resolved sizes)','#D9D9D9')]):
    v=[blk(ds,g)['frac'] for ds in ORDER]
    a.bar(x+(i-1)*wd,v,width=wd,color=col,lw=0,label=lab)
a.set_xticks(x); a.set_xticklabels(ORDER); a.set_ylim(0,1); a.set_ylabel('energy explained by block model')
a.legend(loc='upper right',borderaxespad=0.1,handlelength=1.1,labelspacing=0.25)
# B
a=ax[1]
for ds in ORDER:
    a.plot([0,1],[BP[ds]['PR_full'],BP[ds]['type_side']['PR_block']],'-o',color=DS[ds],ms=4,lw=1.2,label=ds)
a.set_xticks([0,1]); a.set_xticklabels(['full matrix','block model\n(types by side)']); a.set_xlim(-0.3,1.3); a.set_ylabel('effective rank (PR)'); a.set_ylim(0,1300)
a.legend(loc='upper right',borderaxespad=0.1,handlelength=1.4,labelspacing=0.25)
# C
a=ax[2]
for ds in ORDER:
    s=TS[f'{TKEY[ds]}_types|signed_norm']; sn=TS[f'{TKEY[ds]}_types|signed_norm|null_config']
    r=np.arange(1,len(s)+1)/len(s); m=s/s[0]>=1.1e-3; a.plot(r[m],(s/s[0])[m],color=DS[ds],lw=1.1)
    rn=np.arange(1,len(sn)+1)/len(sn); mn_=sn/s[0]>=1.1e-3; a.plot(rn[mn_],(sn/s[0])[mn_],color=DS[ds],lw=0.9,ls=(0,(3,1.5)))   # values below the axis are not drawn
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(8e-5,1); a.set_ylim(1e-3,1.3)
a.set_xlabel('rank / number of types'); a.set_ylabel('singular value / largest')
a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e-4,1e-3,1e-2,1e-1,1])); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:{1e-4:'0.0001',1e-3:'0.001',1e-2:'0.01',1e-1:'0.1',1:'1'}.get(v,''))); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1e-3,1e-2,1e-1,1])); a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:{1e-3:'0.001',1e-2:'0.01',1e-1:'0.1',1:'1'}.get(v,''))); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.legend(handles=[Line2D([],[],color='#4D4D4D',lw=1.1,label='cell-type matrix'),Line2D([],[],color='#4D4D4D',lw=0.9,ls=(0,(3,1.5)),label='degree-preserving null')],loc='lower left',borderaxespad=0.1,handlelength=1.8,labelspacing=0.25)
# D
a=ax[3]; y=np.arange(len(ORDER))[::-1]
for yi,ds in zip(y,ORDER):
    t=TB[f'{TKEY[ds]}_types|signed_norm']; a.plot([t['PR'],t['null_config']['PR']],[yi,yi],color=LIGHT,lw=2.2,zorder=1,solid_capstyle='round')
    a.scatter([t['PR']],[yi],s=24,color=DS[ds],zorder=3); a.scatter([t['null_config']['PR']],[yi],s=24,facecolor='white',edgecolor=DS[ds],lw=1.1,zorder=3)
a.set_yticks(y); a.set_yticklabels([f"{ds}\n{TB[f'{TKEY[ds]}_types|signed_norm']['N']:,} types" for ds in ORDER]); a.tick_params(axis='y',length=0); a.set_xlim(0,1500); a.set_ylim(-0.7,len(ORDER)-0.3)
a.set_xlabel('effective rank of the cell-type matrix (PR)')
a.legend(handles=[Line2D([],[],marker='o',ls='',ms=5,color='#4D4D4D',label='connectome'),Line2D([],[],marker='o',ls='',ms=5,mfc='white',mec='#4D4D4D',label='degree-preserving null')],loc='upper right',bbox_to_anchor=(1.0,1.12),borderaxespad=0.1,handletextpad=0.2,ncol=2,columnspacing=0.8)
label_panels(fig,ax,'ABCD'); gate(fig,'Fig3',ax,'ABCD'); export(fig,'Fig3')
