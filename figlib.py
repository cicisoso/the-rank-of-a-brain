import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:  # optional render-time panel-alignment gate (QA tool); skipped when the module is absent
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    def require_matplotlib_panel_alignment(fig,**kw): print('audit_panel_alignment not installed; alignment gate skipped')
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
plt.rcParams['svg.fonttype']='none'; plt.rcParams['pdf.fonttype']=42
plt.rcParams.update({'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular'})
plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'legend.frameon':False,'lines.linewidth':1.0})
MM=1/25.4
# semantic colours: descending = blue family; ascending = orange family; sensory ascending = teal; sensory descending = violet
C={'DN':'#0F4D92','DN_light':'#7FA6D6','AN':'#D9782D','AN_light':'#F0B98A','SA':'#2E8B8B','SD':'#7B4F9E','EFF':'#4D4D4D','UNANN':'#C9C9C9','ECS':'#FFFFFF','male':'#0F4D92','female':'#B8386F','grey':'#767676','light':'#E6E6E6'}
CLASS_LABEL={'descending_neuron':'Descending (DN)','ascending_neuron':'Ascending (AN)','sensory_ascending':'Sensory ascending (SA)','sensory_descending':'Sensory descending (SD)'}
CLASS_COLOR={'descending_neuron':C['DN'],'ascending_neuron':C['AN'],'sensory_ascending':C['SA'],'sensory_descending':C['SD']}
def panel_label(ax,label,x_pt=-6,y_pt=2):
    from matplotlib.transforms import ScaledTranslation
    off=ScaledTranslation(x_pt/72,y_pt/72,ax.figure.dpi_scale_trans)
    ax.text(0,1,label,transform=ax.transAxes+off,fontsize=9,fontweight='bold',ha='left',va='bottom')
def panel_labels_tight(fig,axes,letters,dx_pt=0,dy_pt=2):
    fig.canvas.draw(); r=fig.canvas.get_renderer()
    for ax,l in zip(axes,letters):
        bb=ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        fig.text(max(0.004,bb.x0+dx_pt/72/fig.get_figwidth()),min(1-11/72/fig.get_figheight(),bb.y1+dy_pt/72/fig.get_figheight()),l,fontsize=9,fontweight='bold',ha='left',va='bottom')
def save(fig,name,**align):
    base=f'figures/{name}'
    require_matplotlib_panel_alignment(fig,json_out=base+'.alignment.json',overlay_svg=base+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,**align)
    return base
def export(fig,base):
    fig.savefig(base+'.pdf'); fig.savefig(base+'.svg'); fig.savefig(base+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'}); fig.savefig(base+'.png',dpi=300); print('saved',base)
def plain_log(ax,axis='both'):
    fmt=matplotlib.ticker.FuncFormatter(lambda v,p: f'{v:g}'); nf=matplotlib.ticker.NullFormatter()
    if axis in ('x','both'): ax.xaxis.set_major_formatter(fmt); ax.xaxis.set_minor_formatter(nf)
    if axis in ('y','both'): ax.yaxis.set_major_formatter(fmt); ax.yaxis.set_minor_formatter(nf)

# project palette (Q10): male = blue, female = magenta, nulls = greys, regions = categorical without red/green pairing
Q={'male':'#0F4D92','male_light':'#7FA6D6','female':'#B8386F','female_light':'#E2A0BE','null':'#8C8C8C','null_light':'#C9C9C9','iid':'#3A3A3A','central':'#D9782D','optic':'#2E8B8B','vnc':'#7B4F9E','ms':'#C9472B','dim':'#E58E5C','iso':'#767676'}
