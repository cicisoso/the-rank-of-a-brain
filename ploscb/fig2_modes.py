"""Fig 2. The leading modes are local circuits, and the effective rank grows with the number of neurons.
A, B  share of the energy (u^2 + v^2) of each of the leading 60 modes in the central brain, optic lobes and nerve cord of the
      MaleCNS (A) and BANC (B) central nervous systems
C     number of neurons carrying each of the leading 200 modes (participation ratio of the squared output loadings)
D     effective rank of random induced subnetworks against their size (mean of three draws), with log-log slopes"""
from figlib_plos import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
SM=load(os.path.join(RES,'scaling_modes.json'))
RC={'central':'#4D4D4D','optic':'#E69F00','vnc':'#56B4E9'}; RL={'central':'central brain','optic':'optic lobes','vnc':'nerve cord'}
fig=plt.figure(figsize=(W,118*MM)); gs=gridspec.GridSpec(2,2,left=0.09,right=0.985,top=0.95,bottom=0.09,wspace=0.3,hspace=0.5)
ax=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
for a,ds in [(ax[0],'MaleCNS'),(ax[1],'BANC')]:
    m=SM['modes'][ds]; sh=np.array(m['share_top100'])[:60]; regs=m['regions']; bottom=np.zeros(len(sh)); x=np.arange(1,len(sh)+1)
    for r in ['central','optic','vnc']:
        if r in regs: v=sh[:,regs.index(r)]; a.bar(x,v,bottom=bottom,width=0.85,color=RC[r],lw=0,label=RL[r]); bottom+=v
    a.set_xlim(0.3,60.7); a.set_ylim(0,1.28); a.set_yticks([0,0.25,0.5,0.75,1]); a.set_xlabel('mode'); a.set_title(f'{ds} CNS')
ax[0].set_ylabel('share of mode energy'); ax[1].legend(loc='upper right',borderaxespad=0.05,handlelength=1.0,labelspacing=0.15,ncol=3,columnspacing=0.8)
# C neurons per mode
a=ax[2]; order=['MaleCNS','FlyWire','BANC','MANC']
data=[np.array(SM['modes'][d]['neurons_out']) for d in order]
bp=a.boxplot(data,positions=np.arange(len(order)),widths=0.5,showfliers=False,whis=(5,95),patch_artist=True,medianprops=dict(color='k',lw=1.2),whiskerprops=dict(lw=0.8),capprops=dict(lw=0.8))
for patch,d in zip(bp['boxes'],order): patch.set_facecolor(DS[d]); patch.set_alpha(0.55); patch.set_edgecolor(DS[d])
a.set_yscale('log'); a.set_xticks(np.arange(len(order))); a.set_xticklabels(order); a.set_ylabel('neurons per mode (leading 200)'); logfmt(a,'y')
a.set_ylim(1,20000)
# D scaling
a=ax[3]
for d in order:
    rows=SM['scaling'][d]['rows']; n=np.array([r['N'] for r in rows]); pr=np.array([r['PR_mean'] for r in rows]); sd=np.array([r['PR_sd'] for r in rows])
    a.errorbar(n,pr,yerr=sd,fmt='-o',ms=3,lw=1.1,color=DS[d],capsize=0,elinewidth=0.8,label=f"{d} (slope {SM['scaling'][d]['slope']:.2f})")
xx=np.array([2500,165000]); a.plot(xx,15*(xx/2500),color=GREY,lw=0.7,ls=':',label='proportional to N')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('neurons in subnetwork, N'); a.set_ylabel('effective rank (PR)'); logfmt(a,'x'); logfmt(a,'y')
a.set_xlim(2000,2.2e5); a.set_ylim(10,4000)
a.legend(loc='lower right',borderaxespad=0.1,handlelength=1.6,labelspacing=0.2)
label_panels(fig,ax,'ABCD'); gate(fig,'Fig2',ax,'ABCD'); export(fig,'Fig2')
