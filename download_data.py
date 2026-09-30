"""Download the public source tables into data/raw (or $FLYBRAIN_DATA). Required for prep.py only; the derived matrices in
data/ are included in the repository, so the analysis and figure scripts run without this step.
  MaleCNS v1.0 (gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/): connectome-weights (1.0 GB), body annotations
  (14 MB), body neurotransmitters (43 MB); with --with-synapses also syn-partners (6.8 GB, needed for the brain/nerve-cord split).
  FlyWire v783 (Codex data release): connections, classification, consolidated_cell_types, neurons (55 MB).
  With --banc-manc (needed only for ploscb/p1_prep.py and ploscb/p6_robustness.py):
  BANC v888 (lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/): meta (57 MB) and the v3 neuron-to-neuron
  edgelist (359 MB); MANC v1.0 (flyem-manc-exports/v1.0/): neuron properties (17 MB) and traced-connections (75 MB).
Usage: python download_data.py [--with-synapses] [--banc-manc]"""
import os, sys, requests
D=os.environ.get('FLYBRAIN_DATA',os.path.join('data','raw')); [os.makedirs(os.path.join(D,s),exist_ok=True) for s in ['flywire','banc','manc']]
FLAT='https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome'; FW='https://storage.googleapis.com/flywire-data/codex/data/fafb/783'
files=[(f'{FLAT}/connectome-weights-male-cns-v1.0-minconf-0.5.feather','connectome-weights-male-cns-v1.0-minconf-0.5.feather'),(f'{FLAT}/body-annotations-male-cns-v1.0-minconf-0.5.feather','body-annotations-male-cns-v1.0-minconf-0.5.feather'),(f'{FLAT}/body-neurotransmitters-male-cns-v1.0.feather','body-neurotransmitters-male-cns-v1.0.feather')]
if '--with-synapses' in sys.argv: files.append((f'{FLAT}/syn-partners-male-cns-v1.0-minconf-0.5.feather','syn-partners-male-cns-v1.0-minconf-0.5.feather'))
files+=[(f'{FW}/{n}',f'flywire/{n}') for n in ['connections.csv.gz','classification.csv.gz','consolidated_cell_types.csv.gz','neurons.csv.gz']]
if '--banc-manc' in sys.argv:
    BANC='https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888'; MANC='https://storage.googleapis.com/flyem-manc-exports/v1.0'
    files+=[(f'{BANC}/{n}',f'banc/{n}') for n in ['banc_888_meta.feather','banc_888_edgelist_simple_v3.feather']]
    files+=[(f'{MANC}/manc-v1.0-neuron-properties.feather','manc/manc-v1.0-neuron-properties.feather'),(f'{MANC}/manc-traced-adjacencies-v1.0/traced-connections.csv','manc/traced-connections.csv')]
for url,name in files:
    dst=os.path.join(D,name)
    if os.path.exists(dst) and os.path.getsize(dst)>0: print('skip',name); continue
    r=requests.get(url,stream=True,timeout=120); r.raise_for_status()
    with open(dst+'.part','wb') as f:
        for ch in r.iter_content(1<<22): f.write(ch)
    os.replace(dst+'.part',dst); print('ok',name)
