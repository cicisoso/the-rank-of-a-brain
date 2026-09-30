"""Figure conventions for PLOS Computational Biology (Python / matplotlib only).
PLOS: TIFF (LZW) or EPS, RGB, 300-600 dpi, width 6.68-19.05 cm, height <= 22.23 cm, Arial/Times/Symbol at 8-12 pt, each figure a
separate file named Fig1.tif ... with no figure number or caption inside the image; panels labelled A, B, C."""
import os, sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    sys.path.insert(0,os.environ.get('NATURE_FIGURE_SCRIPTS',os.path.expanduser('~/.claude/skills/nature-figure/scripts')))
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(fig,**kw): print('audit_panel_alignment not installed; alignment gate skipped')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
    'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular',
    'font.size':8,'axes.labelsize':8.5,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7,'xtick.minor.width':0.5,'ytick.minor.width':0.5,
    'xtick.major.size':3,'ytick.major.size':3,'xtick.minor.size':1.8,'ytick.minor.size':1.8,'legend.frameon':False,'lines.linewidth':1.1,'axes.titlepad':4})
MM=1/25.4; W=190*MM; DPI=450
HERE=os.path.dirname(os.path.abspath(__file__)); Q=os.path.dirname(HERE); RES=os.path.join(HERE,'results'); QRES=os.path.join(Q,'results')
OUT=os.path.join(HERE,'figures'); QA=os.path.join(HERE,'qa'); os.makedirs(OUT,exist_ok=True); os.makedirs(QA,exist_ok=True)
# datasets: males in blues, females in magenta/orange (as in the companion analyses); nulls in greys
DS={'MaleCNS':'#0F4D92','MANC':'#6BAED6','FlyWire':'#B8386F','BANC':'#D9782D'}
NULL={'config':'#4D4D4D','wshuf':'#8C8C8C','er':'#BDBDBD','iid':'#1A1A1A'}
GREY='#767676'; LIGHT='#E6E6E6'
def load(p): return json.load(open(p))
def label_panels(fig,axes,letters,dx_pt=-2,dy_pt=3,size=11):
    fig.canvas.draw(); r=fig.canvas.get_renderer()
    for ax,l in zip(axes,letters):
        bb=ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        fig.text(max(0.003,bb.x0+dx_pt/72/fig.get_figwidth()),min(1-12/72/fig.get_figheight(),bb.y1+dy_pt/72/fig.get_figheight()),l,fontsize=size,fontweight='bold',ha='left',va='bottom')
def gate(fig,name,axes,ids,exemptions=None,groups=None):
    kw=dict(json_out=os.path.join(QA,name+'.alignment.json'),overlay_svg=os.path.join(QA,name+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,axes=axes,panel_ids=list(ids))
    if exemptions: kw['exemptions']=exemptions
    if groups: kw['comparable_groups']=groups
    return require_matplotlib_panel_alignment(fig,**kw)
def export(fig,name):
    base=os.path.join(OUT,name)
    fig.savefig(base+'.pdf'); fig.savefig(base+'.png',dpi=200)
    fig.savefig(base+'.tif',dpi=DPI,pil_kwargs={'compression':'tiff_lzw'})
    from PIL import Image
    im=Image.open(base+'.tif')
    if im.mode!='RGB': im.convert('RGB').save(base+'.tif',compression='tiff_lzw',dpi=(DPI,DPI))
    w,h=fig.get_figwidth()/MM,fig.get_figheight()/MM; assert w<=190.5 and h<=222.3, (w,h)
    print('saved',name,f'{w:.0f} x {h:.0f} mm',round(os.path.getsize(base+'.tif')/1e6,1),'MB')
def logfmt(ax,axis='y'):
    f=matplotlib.ticker.FuncFormatter(lambda v,p: (f'{v:g}' if v<1e4 else f'{v/1e3:g}k'))
    (ax.yaxis if axis=='y' else ax.xaxis).set_major_formatter(f); (ax.yaxis if axis=='y' else ax.xaxis).set_minor_formatter(matplotlib.ticker.NullFormatter())
