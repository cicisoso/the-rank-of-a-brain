import json, glob, numpy as np
S={}; Z={}
for f in sorted(f for f in glob.glob('results/spectrum_*.json') if 'summary' not in f):
    S.update(json.load(open(f)))
for f in sorted(glob.glob('results/spectra_*.npz')):
    z=np.load(f); Z.update({k:z[k] for k in z.files})
json.dump(S,open('results/spectrum_summary.json','w'),indent=1); np.savez_compressed('results/spectra.npz',**Z); print('merged',len(S),'summaries',len(Z),'spectra')
