"""Fig 1. Four complete connectomes have an effective rank far below their size and below degree-matched random networks.
A  leading singular values of the MaleCNS central nervous system and of three nulls with the same weighted edges
B  energy captured by the leading modes of the four whole connectomes and of their degree-preserving nulls
C  effective rank (participation ratio, 1,024-probe estimate) of every network and region, against its degree-preserving null
D  ratio of null to connectome effective rank for synapse counts and for signed, input-normalised weights"""
from figlib_plos import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
Z=np.load(os.path.join(QRES,'spectra.npz')); SQ=load(os.path.join(QRES,'spectrum_summary.json'))
ZB=np.load(os.path.join(RES,'spectra_banc_cns.npz')); SB=load(os.path.join(RES,'spectrum_banc_cns.json'))
ZM=np.load(os.path.join(RES,'spectra_manc_all.npz')); SM=load(os.path.join(RES,'spectrum_manc_all.json'))
PT=load(os.path.join(RES,'pr_table.json'))
def curve(S,f2): return np.cumsum(S**2)/f2
WHOLE={'MaleCNS':(Z['male_cns|count'],SQ['male_cns|count'],Z['male_cns|count|null_config'],SQ['male_cns|count']['null_config']),
       'FlyWire':(Z['female_brain|count'],SQ['female_brain|count'],Z['female_brain|count|null_config'],SQ['female_brain|count']['null_config']),
       'BANC':(ZB['banc_cns|count|real'],SB['banc_cns|count'],ZB['banc_cns|count|config'],SB['banc_cns|count']['null_config']),
       'MANC':(ZM['manc_all|count|real'],SM['manc_all|count'],ZM['manc_all|count|config'],SM['manc_all|count']['null_config'])}
fig=plt.figure(figsize=(W,150*MM)); gs=gridspec.GridSpec(2,3,left=0.185,right=0.985,top=0.955,bottom=0.075,wspace=0.5,hspace=0.42,height_ratios=[1,1.15])
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axD=fig.add_subplot(gs[0,2]); axC=fig.add_subplot(gs[1,:])
ax=[axA,axB,axC,axD]
# A
a=ax[0]; s=Z['male_cns|count']; s0=s[0]
a.plot(np.arange(1,len(s)+1),s/s0,color=DS['MaleCNS'],lw=1.4,label='MaleCNS CNS')
for nm,lab in [('config','degree-preserving'),('wshuf','weight-shuffled'),('er','random placement')]:
    sn=Z[f'male_cns|count|null_{nm}']; a.plot(np.arange(1,len(sn)+1),sn/s0,color=NULL[nm],lw=1.1,label=lab)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(0.8,2500); a.set_ylim(0.02,1.2); a.set_xlabel('mode'); a.set_ylabel('singular value / largest')
for axis in (a.xaxis,a.yaxis): axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}')); axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.02,0.05,0.1,0.2,0.5,1]))
a.legend(loc='lower left',borderaxespad=0.1,handlelength=1.6,labelspacing=0.2)
# B
a=ax[1]
for ds,(S,res,Sn,rn) in WHOLE.items():
    a.plot(np.arange(1,len(S)+1),curve(S,res['frob2']),color=DS[ds],lw=1.3,label=ds)
    cn=np.cumsum(Sn**2); cn=cn/cn[-1]*rn['energy_topk']; a.plot(np.arange(1,len(cn)+1),cn,color=DS[ds],lw=0.9,ls=(0,(3,1.5)))
a.set_xscale('log'); a.set_xlim(0.8,2500); a.set_ylim(0,0.9); a.set_xlabel('leading modes, k'); a.set_ylabel('fraction of energy (Σσ²)')
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}')); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.legend(handles=[Line2D([],[],color=DS[d],lw=1.3,label=d) for d in WHOLE]+[Line2D([],[],color='#4D4D4D',lw=0.9,ls=(0,(3,1.5)),label='null')],loc='upper left',borderaxespad=0.1,handlelength=1.6,labelspacing=0.15)
# C
a=ax[2]; rows=[r for r in PT if r['flavour']=='count' and r['region']!='nerve cord (all traced)']
y=np.arange(len(rows))[::-1]
for yi,r in zip(y,rows):
    a.plot([r['PR'],r['PR_null']],[yi,yi],color=LIGHT,lw=2.2,zorder=1,solid_capstyle='round')
    a.scatter([r['PR']],[yi],s=20,color=DS[r['dataset']],zorder=3); a.scatter([r['PR_null']],[yi],s=20,facecolor='white',edgecolor=DS[r['dataset']],lw=1.0,zorder=3)
a.set_xscale('log'); a.set_xlim(150,30000); a.set_yticks(y); a.set_yticklabels([f"{r['dataset']} {r['region']}" for r in rows]); a.tick_params(axis='y',length=0); a.set_ylim(-0.6,len(rows)-0.4)
a.set_xlabel('effective rank (PR)'); a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}')); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([200,500,1000,2000,5000,10000,20000]))
a.legend(handles=[Line2D([],[],marker='o',ls='',ms=5,color='#4D4D4D',label='connectome'),Line2D([],[],marker='o',ls='',ms=5,mfc='white',mec='#4D4D4D',label='degree-preserving null')],loc='lower right',borderaxespad=0.1,handletextpad=0.2,labelspacing=0.2)
# D
a=ax[3]; whole=[('MaleCNS','CNS'),('FlyWire','brain'),('BANC','CNS'),('MANC','nerve cord (all traced)')]; x=np.arange(len(whole)); wd=0.36
for i,(flav,lab,hatch_col) in enumerate([('count','synapse counts','#4D4D4D'),('signed_norm','signed, input-normalised','#BDBDBD')]):
    v=[next(r for r in PT if r['dataset']==d and r['region']==g and r['flavour']==flav)['null_over_real'] for d,g in whole]
    a.bar(x+(i-0.5)*wd,v,width=wd,color=hatch_col,lw=0,label=lab)
a.axhline(1,color=GREY,lw=0.6,ls=':'); a.set_xticks(x); a.set_xticklabels(['Male\nCNS','Fly\nWire','BANC','MANC'],fontsize=8); a.set_ylim(0,15); a.set_ylabel('null effective rank / connectome')
a.legend(loc='upper right',borderaxespad=0.0,handlelength=1.0,labelspacing=0.15,fontsize=8)
label_panels(fig,ax,'ABCD'); gate(fig,'Fig1',[axA,axB,axD],'ABD'); export(fig,'Fig1')
