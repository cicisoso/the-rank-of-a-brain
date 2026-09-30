"""Supporting figures (PLOS: S1 Fig, S2 Fig, S3 Fig), one file each.
S1  estimator validation: Hutchinson participation ratio against exact dense SVD on random 8,000-neuron subnetworks; accuracy of
    the randomized SVD; relative standard error of every 1,024-probe estimate
S2  robustness: effective rank and null/connectome ratio across connection thresholds and weight transforms
S3  cross-animal comparison in full: relative capture against k, nerve-cord difference spectra, and the cell classes carrying the
    leading male-female difference mode"""
from figlib_plos import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
RB=load(os.path.join(RES,'robustness.json')); PT=load(os.path.join(RES,'pr_table.json')); SI=load(os.path.join(RES,'sex_individual.json'))
SS=np.load(os.path.join(RES,'sex_individual_spectra.npz')); SMODE=load(os.path.join(RES,'sex_mode.json'))
ORDER=['MaleCNS','FlyWire','BANC','MANC']
def fin(fig,ax,name,ids,ex=None): label_panels(fig,ax,ids); gate(fig,name,ax,ids,exemptions=ex); export(fig,name); plt.close(fig)
# ---------------- S1
fig=plt.figure(figsize=(W,62*MM)); gs=gridspec.GridSpec(1,3,left=0.085,right=0.985,top=0.9,bottom=0.23,wspace=0.45); ax=[fig.add_subplot(gs[0,j]) for j in range(3)]
a=ax[0]; col={'MaleCNS central':DS['MaleCNS'],'FlyWire central':DS['FlyWire'],'MANC':DS['MANC']}
for k,rows in RB['B'].items():
    a.errorbar([r['PR_exact'] for r in rows],[r['PR_hutch_mean'] for r in rows],yerr=[r['PR_hutch_sd'] for r in rows],fmt='o',ms=4,color=col[k],capsize=0,elinewidth=0.8,label=k)
a.plot([90,310],[90,310],color=GREY,lw=0.6,ls='--'); a.set_xlim(90,310); a.set_ylim(90,310); a.set_xlabel('exact PR (dense SVD)'); a.set_ylabel('Hutchinson PR (48 probes)')
a.legend(loc='upper left',borderaxespad=0.1,handletextpad=0.2,labelspacing=0.2)
a=ax[1]; xs=[]
for i,(k,rows) in enumerate(RB['B'].items()):
    a.scatter([i]*len(rows),[r['rsvd_max_rel_err_top200']*100 for r in rows],s=18,color=col[k],zorder=3)
a.set_xticks(range(3)); a.set_xticklabels(['MaleCNS','FlyWire','MANC']); a.set_ylim(0,0.6); a.set_ylabel('max. error of top 200 σ (%)'); a.set_xlim(-0.5,2.5)
a=ax[2]; rows=[r for r in PT if r['flavour']=='count']; rs=[r for r in PT if r['flavour']=='signed_norm']
a.scatter(np.arange(len(rows)),[r['rel_se']*100 for r in rows],s=16,color='#4D4D4D',label='synapse counts',zorder=3)
a.scatter(np.arange(len(rs)),[r['rel_se']*100 for r in rs],s=16,facecolor='white',edgecolor='#4D4D4D',lw=1.0,label='signed, normalised',zorder=3)
a.set_xticks([]); a.set_xlabel('network (15 per weight type)'); a.set_ylabel('relative s.e. of PR (%)'); a.set_ylim(0,4.4); a.set_yticks([0,1,2,3])
a.legend(loc='upper left',borderaxespad=0.1,handletextpad=0.2,labelspacing=0.2)
fin(fig,ax,'S1_Fig','ABC')
# ---------------- S2
fig=plt.figure(figsize=(W,68*MM)); gs=gridspec.GridSpec(1,2,left=0.085,right=0.985,top=0.9,bottom=0.25,wspace=0.3); ax=[fig.add_subplot(gs[0,j]) for j in range(2)]
TAGS=['count>=1','count>=3','count>=5','count>=10','binary>=5','log>=5']; LAB=['≥1','≥3','≥5','≥10','binary\n≥5','log\n≥5']
for a,key,yl in [(ax[0],'PR','effective rank (PR)'),(ax[1],'null_over_real','null / connectome effective rank')]:
    for ds in ORDER:
        v=[RB['A'][ds][t][key] for t in TAGS]; a.plot(range(len(TAGS)),v,'-o',ms=3.5,lw=1.1,color=DS[ds],label=ds)
    a.set_xticks(range(len(TAGS))); a.set_xticklabels(LAB); a.set_xlabel('synapses per connection (threshold) and weight'); a.set_ylabel(yl)
ax[0].set_yscale('log'); logfmt(ax[0],'y'); ax[0].set_ylim(300,8000); ax[1].axhline(1,color=GREY,lw=0.6,ls=':'); ax[1].set_ylim(0,6)
ax[0].legend(loc='upper left',borderaxespad=0.1,ncol=2,columnspacing=0.8,handlelength=1.4,labelspacing=0.2)
fin(fig,ax,'S2_Fig','AB')
# ---------------- S3
PAIRS={'brain':[('MaleCNS','FlyWire'),('FlyWire','BANC'),('MaleCNS','BANC')],'vnc':[('MaleCNS','MANC'),('MaleCNS','BANC'),('MANC','BANC')]}
KCOL={'within animal':'#8C8C8C','same sex':'#1A1A1A','different sex':'#B2182B'}
fig=plt.figure(figsize=(W,120*MM)); gs=gridspec.GridSpec(2,2,left=0.09,right=0.985,top=0.95,bottom=0.09,wspace=0.3,hspace=0.5); ax=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
for a,reg,title in [(ax[0],'brain','brain'),(ax[1],'vnc','nerve cord')]:
    d=SI[reg]; k=d['k']; cap=d['capture']
    for nm in d['animals']:
        v=np.mean([cap[f'{nm}_L|{nm}_R']['rel'],cap[f'{nm}_R|{nm}_L']['rel']],axis=0); a.plot(k,v,'-o',ms=2.5,lw=1.0,color=DS[nm])
    for x,y in PAIRS[reg]:
        v=np.mean([cap[f'{p}_{s}|{q}_{s}']['rel'] for p,q in [(x,y),(y,x)] for s in 'LR'],axis=0); kind=cap[f'{x}|{y}']['kind']
        a.plot(k,v,'-s',ms=2.5,lw=1.2,color=KCOL[kind])
    a.set_xscale('log'); a.set_ylim(0,1.45); a.set_yticks([0,0.25,0.5,0.75,1]); a.set_xlabel('leading modes, k'); a.set_ylabel('relative capture'); a.set_title(title)
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}')); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(handles=[Line2D([],[],color='#4D4D4D',marker='o',ms=2.5,lw=1.0,label='left vs right (colour = animal)'),Line2D([],[],color=KCOL['same sex'],marker='s',ms=2.5,lw=1.2,label='same sex, two animals'),Line2D([],[],color=KCOL['different sex'],marker='s',ms=2.5,lw=1.2,label='different sex')],loc='upper left',borderaxespad=0.05,handlelength=1.8,labelspacing=0.15)
a=ax[2]
for nm in SI['vnc']['animals']:
    s=SS[f'vnc|diff|{nm}_L-{nm}_R'][:40]; a.plot(np.arange(1,len(s)+1),s,color=KCOL['within animal'],lw=1.0)
for x,y in PAIRS['vnc']:
    s=SS[f'vnc|diff|{x}_L-{y}_L'][:40]; a.plot(np.arange(1,len(s)+1),s,color=KCOL[SI['vnc']['capture'][f'{x}|{y}']['kind']],lw=1.2)
a.set_xscale('log'); a.set_xlim(0.9,45); a.set_ylim(0,4); a.set_xlabel('mode of the difference matrix'); a.set_ylabel('singular value'); a.set_title('nerve cord, left hemispheres')
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}')); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a=ax[3]; keys=['MaleCNS-FlyWire L','MaleCNS-FlyWire R','MaleCNS-BANC L','MaleCNS-BANC R']; cats=['cb_sensory','cb_intrinsic','other']; cc={'cb_sensory':'#2E8B8B','cb_intrinsic':'#4D4D4D','other':'#BDBDBD'}
bottom=np.zeros(len(keys))
for c in cats:
    v=[]
    for k_ in keys:
        sc=SMODE[k_]['superclass_of_top1pct']; tot=sum(sc.values()); v.append((sc.get(c,0) if c!='other' else tot-sc.get('cb_sensory',0)-sc.get('cb_intrinsic',0))/tot)
    a.bar(range(len(keys)),v,bottom=bottom,color=cc[c],width=0.6,lw=0,label={'cb_sensory':'central-brain sensory','cb_intrinsic':'central-brain intrinsic','other':'other'}[c]); bottom+=np.array(v)
a.set_xticks(range(len(keys))); a.set_xticklabels(['vs FlyWire\nleft','vs FlyWire\nright','vs BANC\nleft','vs BANC\nright']); a.set_ylim(0,1.35); a.set_yticks([0,0.25,0.5,0.75,1])
a.set_ylabel('share of top 1% types'); a.set_title('leading mode of MaleCNS − female')
a.legend(loc='upper center',borderaxespad=0.1,ncol=2,handlelength=1.0,labelspacing=0.15,columnspacing=0.8)
fin(fig,ax,'S3_Fig','ABCD')
