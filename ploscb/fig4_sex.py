"""Fig 4. Among shared cell types, animals of different sex differ no more than animals of the same sex.
A, B  relative capture (mean over k = 10-400 leading modes) of one hemisphere's shared-type matrix by the modes of another:
      left vs right within each animal, and the same hemisphere of two animals of the same or different sex
C     entrywise similarity of the same hemisphere in two animals, corrected for each animal's left-right similarity
D     singular values of difference matrices in the brain: left minus right within animals, and the same hemisphere of two animals"""
from figlib_plos import *
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
SI=load(os.path.join(RES,'sex_individual.json')); SIM=load(os.path.join(RES,'similarity.json')); SS=np.load(os.path.join(RES,'sex_individual_spectra.npz'))
PAIRS={'brain':[('MaleCNS','FlyWire'),('FlyWire','BANC'),('MaleCNS','BANC')],'vnc':[('MaleCNS','MANC'),('MaleCNS','BANC'),('MANC','BANC')]}
KCOL={'within animal':'#8C8C8C','same sex':'#1A1A1A','different sex':'#B2182B'}
fig=plt.figure(figsize=(W,122*MM)); gs=gridspec.GridSpec(2,2,left=0.215,right=0.985,top=0.95,bottom=0.09,wspace=0.66,hspace=0.45)
ax=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
def capture_panel(a,reg,title):
    d=SI[reg]; cap=d['capture']; names=list(d['animals']); rows=[]
    for nm in names: rows.append((f'{nm} left vs right','within animal',[np.mean(cap[f'{nm}_L|{nm}_R']['rel']),np.mean(cap[f'{nm}_R|{nm}_L']['rel'])]))
    for x,y in PAIRS[reg]:
        kind=cap[f'{x}|{y}']['kind']; rows.append((f'{x} vs {y}',kind,[np.mean(cap[f'{p}_{s}|{q}_{s}']['rel']) for p,q in [(x,y),(y,x)] for s in 'LR']))
    yy=np.arange(len(rows))[::-1]
    for yi,(lab,kind,v) in zip(yy,rows):
        a.plot([min(v),max(v)],[yi,yi],color=LIGHT,lw=2.2,zorder=1,solid_capstyle='round'); a.scatter(v,[yi]*len(v),s=16,color=KCOL[kind],zorder=3)
    a.set_yticks(yy); a.set_yticklabels([r[0] for r in rows]); a.tick_params(axis='y',length=0); a.set_xlim(0,1); a.set_ylim(-0.7,len(rows)-0.3)
    a.set_xlabel('relative capture (mean over k)'); a.set_title(title)
capture_panel(ax[0],'brain','brain, 6,652 shared types'); capture_panel(ax[1],'vnc','nerve cord, 3,232 shared types')
ax[1].legend(handles=[Line2D([],[],marker='o',ls='',ms=5,color=c,label=k) for k,c in KCOL.items()],loc='lower right',borderaxespad=0.1,handletextpad=0.2,labelspacing=0.2)
# C corrected similarity
a=ax[2]; rows=[]
for reg,lab in [('brain','brain'),('vnc','cord')]:
    for x,y in PAIRS[reg]:
        rr=[r for r in SIM[reg]['rows'] if r['pair'].startswith(f'{x} vs {y}')]; rows.append((f'{x} vs {y} ({lab})',[r['r_corrected'] for r in rr],rr[0]['kind']))
yy=np.arange(len(rows))[::-1]
for yi,(lab,v,kind) in zip(yy,rows):
    a.plot([min(v),max(v)],[yi,yi],color=LIGHT,lw=2.2,zorder=1,solid_capstyle='round'); a.scatter(v,[yi]*len(v),s=16,color=KCOL[kind],zorder=3)
a.set_yticks(yy); a.set_yticklabels([r[0] for r in rows]); a.tick_params(axis='y',length=0); a.set_xlim(0.5,1.0); a.set_ylim(-0.7,len(rows)-0.3)
a.set_xlabel('similarity corrected for left–right reliability')
# D difference spectra (brain)
a=ax[3]
for nm in SI['brain']['animals']:
    s=SS[f'brain|diff|{nm}_L-{nm}_R'][:40]; a.plot(np.arange(1,len(s)+1),s,color=KCOL['within animal'],lw=1.0)
for x,y in PAIRS['brain']:
    s=SS[f'brain|diff|{x}_L-{y}_L'][:40]; kind=SI['brain']['capture'][f'{x}|{y}']['kind']; a.plot(np.arange(1,len(s)+1),s,color=KCOL[kind],lw=1.2)
a.set_xscale('log'); a.set_xlim(0.9,45); a.set_ylim(0,5.5); a.set_xlabel('mode of the difference matrix'); a.set_ylabel('singular value'); a.set_title('brain, left hemispheres')
a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}')); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.legend(handles=[Line2D([],[],color=KCOL['within animal'],lw=1.0,label='left − right (3 animals)'),Line2D([],[],color=KCOL['same sex'],lw=1.2,label='FlyWire − BANC'),Line2D([],[],color=KCOL['different sex'],lw=1.2,label='MaleCNS − FlyWire or BANC')],loc='upper right',borderaxespad=0.1,handlelength=1.6,labelspacing=0.2)
label_panels(fig,ax,'ABCD'); gate(fig,'Fig4',ax,'ABCD'); export(fig,'Fig4')
