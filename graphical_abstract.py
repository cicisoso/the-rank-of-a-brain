"""Graphical abstract for Cell Systems: exact square, 1200 x 1200 px at 300 dpi (4 x 4 inches)."""
from figlib import *
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
import json
from matplotlib.patches import Rectangle, FancyArrowPatch
S=json.load(open('results/spectrum_summary.json')); Z=np.load('results/spectra.npz')
fig=plt.figure(figsize=(4,4),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
ax.text(50,95,'What is the rank of a brain?',ha='center',va='center',fontsize=13,fontweight='bold')
# left: matrix icon with a border patch
ax.add_patch(Rectangle((6,45),34,34,facecolor='#E6E6E6',edgecolor='k',lw=0.8)); ax.add_patch(Rectangle((40,45),5,34,facecolor=Q['ms'],edgecolor='k',lw=0.8)); ax.add_patch(Rectangle((6,40),39,5,facecolor=Q['ms'],edgecolor='k',lw=0.8))
ax.text(23,62,'164,606\nneurons',ha='center',va='center',fontsize=8); ax.text(25,36,'Male-specific neurons:\na patch of rank ~100',ha='center',va='top',fontsize=7,color=Q['ms'])
ax.text(23,82,'Complete male CNS',ha='center',va='bottom',fontsize=8)
# right: spectrum inset
ins=fig.add_axes([0.56,0.42,0.40,0.36]); s=Z['male_cns|count']; sn=Z['male_cns|count|null_config']
ins.plot(np.arange(1,len(s)+1),s/s[0],color=Q['male'],lw=1.4,label='Connectome'); ins.plot(np.arange(1,len(sn)+1),sn/s[0],color=Q['null'],lw=1.1,ls='--',label='Degree-matched\nrandom')
ins.set_xscale('log'); ins.set_yscale('log'); plain_log(ins); ins.set_xlabel('Mode rank',fontsize=7); ins.set_ylabel('Singular value',fontsize=7); ins.tick_params(labelsize=6); ins.legend(fontsize=6,loc='lower left',handlelength=1.4); ins.set_ylim(0.05,2)
ins.set_title(f"Effective rank {S['male_cns|count']['PR']:.0f} of 164,606",fontsize=8)
# bottom: three conclusions
for x,txt in [(17,'Rank ≈ 1,000\n0.7% of N, 4× below\nrandom wiring'),(50,'Set by cell types\nblock model explains\n2/3 of connectivity'),(83,'Sex adds neurons,\nnot dimensions\n(low-rank patch)')]:
    ax.add_patch(Rectangle((x-15,4),30,20,facecolor='#F4F4F4',edgecolor='#BBBBBB',lw=0.8)); ax.text(x,14,txt,ha='center',va='center',fontsize=7.2)
ax.annotate('',xy=(50,28),xytext=(50,33),arrowprops=dict(arrowstyle='-|>',lw=0.8,color='k'))
fig.savefig('figures/Graphical_Abstract.png',dpi=300); fig.savefig('figures/Graphical_Abstract.tiff',dpi=300,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig('figures/Graphical_Abstract.pdf')
from PIL import Image; im=Image.open('figures/Graphical_Abstract.png'); print('GA size',im.size)
