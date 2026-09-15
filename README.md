# The rank of a brain: cell types set the dimensionality of the fly connectome, and sex adds a low-rank patch

Code, derived connectivity matrices, spectra and figures for the manuscript *"The rank of a brain: cell types set the dimensionality of the fly connectome, and sex adds a low-rank patch"* (Liu, Zong, Chen and Xiong; School of Information Technology, Zhejiang Financial College; submitted to *Cell Systems* as a Report).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22760814.svg)](https://doi.org/10.5281/zenodo.22760814)

We measure the singular value spectra of two complete *Drosophila* connectomes, the MaleCNS v1.0 male central nervous system (164,606 neurons) and the FlyWire v783 female brain (139,255 neurons), against degree-preserving, weight-shuffled and Erdős–Rényi null models. Both have an effective rank (participation ratio) near one thousand; the leading modes are local circuits; the rank grows with size as N^0.8; a block model constant within cell-type pairs explains two-thirds of the connectivity and the cell-type matrix has the spectrum of a degree-matched random graph; male and female brains share their leading dimensions; and the 1,365 male-specific neurons form a patch of effective rank about 100 that lies inside the shared subspace.

## Contents

| Path | What it holds |
|---|---|
| `*.py` | Analysis and figure scripts (Python 3.11), run from the repository root. |
| `data/` | Derived sparse matrices (synapse counts, pairs with ≥5 synapses; SciPy `.npz`) and node tables (`.parquet`) for the male CNS, male brain-only, male nerve-cord-only and female brain, built by `prep.py`. |
| `results/` | Spectra (`spectra.npz`, `type_spectra.npz`, `alignment.npz`), summaries (`spectrum_summary.json`, `signal_rank.json`, `types_sex.json`, `sex_followup.json`, `male_specific.json`, `type_redundancy.json`), containment scores and mode localisation. |
| `figures/` | Figures 1–4, Figure S1 and the graphical abstract (PDF and PNG). |
| `tables/` | Table S1 (`Table_S1.xlsx`, 12 sheets). |

## Data sources (public; raw files not included)

- MaleCNS v1.0 (Berg et al., 2026; CC-BY 4.0), bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (neuron-to-neuron synapse counts), `body-annotations-…feather`, `body-neurotransmitters-…feather`, and `syn-partners-…feather` (synapse table with neuropil labels; only needed to rebuild the brain-only and nerve-cord-only matrices).
- FlyWire FAFB v783 (Dorkenwald et al., 2024; Schlegel et al., 2024), Codex data release: `connections.csv.gz`, `classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz`.

`python download_data.py [--with-synapses]` fetches these into `data/raw/` (or `$FLYBRAIN_DATA`). The derived matrices in `data/` are included, so every step after `prep.py` runs without the raw download.

## Setup

```bash
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=2      # the spectrum jobs are run as one process per matrix
```

## Pipeline (run from the repository root)

1. `download_data.py`, then `prep.py` – build the sparse matrices and node tables in `data/` (about 5 minutes; the synapse-table pass needs ~30 GB RAM).
2. `spectrum_main.py <matrix>` for each of `male_cns male_brain male_central male_optic male_vnc female_brain female_central female_optic` (run in parallel; 20–60 min each) – participation ratios, leading singular values (randomized SVD) and null models; `null_signed.py male_cns` / `female_brain` – renormalised nulls for the signed, input-normalised flavour; `merge_spectra.py` – merge per-matrix outputs into `results/spectrum_summary.json` and `results/spectra.npz`; `signal_rank.py` – modes above each null's spectral edge.
3. `types_sex.py` – full spectra of cell-type matrices, shared-type sex comparison, alignment, cross-projection, difference-matrix rank and fru/dsx loading; `sex_followup.py` – hemisphere-level cross-projection references.
4. `male_specific.py` – removal, containment and patch-rank tests, rank scaling with network size, mode composition and localisation.
5. `type_redundancy.py` – least-squares block model by cell type and cell type × hemisphere.
6. `fig1_spectrum.py` … `fig4_malespecific.py`, `figS1_controls.py`, `graphical_abstract.py` – figures (`figlib.py` and `speclib.py` hold shared styling and spectral helpers; the optional panel-alignment gate runs only if the `audit_panel_alignment` module is on the path).

`speclib.py` implements the exact Frobenius energy, the Hutchinson estimate of Σσ⁴ (participation ratio), randomized SVD, input normalisation and neurotransmitter signing, and the three null models.

## Citation

Please cite the paper (reference to be added on publication) and the Zenodo archive of this repository (https://doi.org/10.5281/zenodo.22760814), together with the MaleCNS (Berg et al., 2026) and FlyWire (Dorkenwald et al., 2024; Schlegel et al., 2024) connectome papers.

## License

Code: MIT License. Derived matrices, results and figures: CC BY 4.0 (derived from CC-BY 4.0 source data).
