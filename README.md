# The rank of a brain: cell types set the effective rank of complete *Drosophila* connectomes

Code, derived connectivity matrices, spectra and figures for the manuscript *"Cell types set the effective rank of four complete Drosophila connectomes"* (Liu, Zong, Chen and Xiong; School of Information Technology, Zhejiang Financial College; submitted to *PLOS Computational Biology*).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23088621.svg)](https://doi.org/10.5281/zenodo.23088621)

We measure the singular value spectra of four complete *Drosophila* connectomes: MaleCNS v1.0 (male central nervous system, 164,606 neurons), FlyWire v783 (female brain, 139,255 neurons), BANC v888 (female brain and nerve cord, 146,404 neurons) and MANC v1.0 (male nerve cord, 23,193 traced neurons). Their effective ranks (participation ratios) are 1,112, 781, 845 and 461, which is 0.6–2.0% of the number of neurons and 2.4–4.5 times lower than in degree-preserving random networks. The leading modes are local circuits of a few hundred neurons, and the effective rank grows with the number of neurons. A block model that is constant within pairs of cell types split by side explains 42–71% of the connectivity, and the cell-type matrices have spectra comparable to those of degree-matched random graphs. On shared cell types, and after correction for each dataset's left–right reliability, the MaleCNS and FlyWire brains are about as similar as two female brains, apart from one mode in the olfactory input. Neurons present in only one sex take their inputs from dimensions that both sexes share.

## Versions

- **v2.0.0** (this version, https://doi.org/10.5281/zenodo.23088621; PLOS Computational Biology submission): adds `ploscb/`, which extends every analysis to BANC and MANC, recomputes all effective ranks with 1,024 Hutchinson probes, validates the estimators against exact SVD, tests thresholds and weightings, compares sex with individual variation in three brains and three nerve cords, and tests sex-specific neurons of both sexes against 50 matched random draws. It also holds the figures and S1 Table of the current manuscript.
- **v1.0.0** (https://doi.org/10.5281/zenodo.22760814): the two-connectome analyses of an earlier version of the manuscript (the top-level scripts, `data/`, `results/`, `figures/`, `tables/`). They are kept unchanged because `ploscb/` reuses their matrices and spectra.

Two statements of v1.0.0 were revised in v2.0.0. (i) The 48-probe participation ratios of `spectrum_main.py` have standard errors of 3–12% (the earlier manuscript stated below 1%); the current values are the 1,024-probe estimates in `ploscb/results/pr_table.json` (MaleCNS 1,112 ± 8, FlyWire 781 ± 5). (ii) With 50 matched random draws instead of 10, the lower effective rank of the male-specific patch is not significant (p = 0.06 for the brain, 0.08 for the CNS), and female-specific neurons show no such difference (`ploscb/results/patch_*.json`).

## Contents

| Path | What it holds |
|---|---|
| `*.py` (top level) | v1.0.0 analysis and figure scripts (Python 3.11), run from the repository root; `speclib.py` holds the spectral helpers used by all scripts. |
| `data/` | Derived sparse matrices (synapse counts, pairs with ≥5 synapses; SciPy `.npz`) and node tables (`.parquet`) for the male CNS, male brain-only, male nerve-cord-only and female brain, built by `prep.py`. |
| `results/` | v1.0.0 spectra and summaries (`spectra.npz`, `spectrum_summary.json`, `signal_rank.json`, `types_sex.json`, `sex_followup.json`, `male_specific.json`, `type_redundancy.json`, containment scores, mode localisation). `ploscb/` reads `spectra.npz`, `spectrum_summary.json` and `signal_rank.json`. |
| `figures/`, `tables/` | v1.0.0 figures and Table S1. |
| `ploscb/` | v2.0.0 analyses of four connectomes (see below): scripts, derived BANC and MANC matrices (`ploscb/data/`), results (`ploscb/results/`), Figs 1–5 and S1–S3 Figs (`ploscb/figures/`, PDF and PNG) and S1 Table (`ploscb/tables/S1_Table.xlsx`). |

## Data sources (public; raw files not included)

- MaleCNS v1.0 (Berg et al., 2026; CC BY 4.0), bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`: `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (neuron-to-neuron synapse counts), `body-annotations-…feather`, `body-neurotransmitters-…feather`, and `syn-partners-…feather` (synapse table with neuropil labels; only needed to rebuild the brain-only and nerve-cord-only matrices).
- FlyWire FAFB v783 (Dorkenwald et al., 2024; Schlegel et al., 2024), Codex data release: `connections.csv.gz`, `classification.csv.gz`, `consolidated_cell_types.csv.gz`, `neurons.csv.gz`.
- BANC v888 (Bates et al., 2026; CC BY 4.0; Harvard Dataverse https://doi.org/10.7910/DVN/7WTH1N), bucket `lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/`: `banc_888_meta.feather` and the v3 neuron-to-neuron edgelist `banc_888_edgelist_simple_v3.feather`.
- MANC v1.0 (Takemura et al., 2024; Marin et al., 2024; CC BY 4.0), bucket `flyem-manc-exports/v1.0/`: `manc-v1.0-neuron-properties.feather` and `manc-traced-adjacencies-v1.0/traced-connections.csv`.

`python download_data.py [--with-synapses] [--banc-manc]` fetches these into `data/raw/` (or `$FLYBRAIN_DATA`). The derived matrices in `data/` and `ploscb/data/` are included, so every step after the two preparation scripts runs without the raw download, except `ploscb/p4*.py` (which read MaleCNS nerve-cord type names from the MaleCNS body annotations) and `ploscb/p6_robustness.py` (which rebuilds the matrices at other thresholds from the raw edge tables).

## Setup

```bash
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=2      # the spectrum jobs are run as one process per matrix
export FLYBRAIN_DATA=data/raw      # where download_data.py put the raw tables
```

## v1.0.0 pipeline (two connectomes; run from the repository root)

1. `download_data.py`, then `prep.py`: build the sparse matrices and node tables in `data/` (about 5 minutes; the synapse-table pass needs about 30 GB RAM).
2. `spectrum_main.py <matrix>` for each of `male_cns male_brain male_central male_optic male_vnc female_brain female_central female_optic` (run in parallel; 20–60 min each): participation ratios, leading singular values (randomized SVD) and null models. `null_signed.py male_cns` and `null_signed.py female_brain`: renormalised nulls for the signed, input-normalised weights. `merge_spectra.py`: merge per-matrix outputs into `results/spectrum_summary.json` and `results/spectra.npz`. `signal_rank.py`: modes above each null's spectral edge.
3. `types_sex.py`: cell-type matrices and the shared-type sex comparison; `sex_followup.py`: hemisphere-level references.
4. `male_specific.py`: removal, containment and patch-rank tests (10 matched draws), rank scaling and mode localisation.
5. `type_redundancy.py`: least-squares block model by cell type and by cell type × side.
6. `fig1_spectrum.py` … `fig4_malespecific.py`, `figS1_controls.py`, `graphical_abstract.py`: v1.0.0 figures.

## v2.0.0 pipeline (four connectomes; `ploscb/`)

The scripts locate their inputs relative to their own file, so they can be run from any directory, for example `python ploscb/p3_types.py`. They import `speclib.py` from the repository root and read the v1.0.0 matrices in `data/`.

1. `python download_data.py --banc-manc`, then `ploscb/p1_prep.py`: BANC and MANC node tables and matrices in `ploscb/data/` (pairs with ≥5 synapses). MANC neurons receive the MaleCNS nerve-cord type name through the MaleCNS `mancGroup` field, which equals the MANC `group` identifier.
2. `ploscb/p2_spectra.py <network>` for each of `banc_cns banc_brain banc_central banc_optic banc_vnc manc_all manc_vnc`: participation ratio, leading singular values and the three null models, as in `spectrum_main.py`.
3. `ploscb/p3_types.py`: block model for BANC and MANC, and spectra of the cell-type matrices of all four connectomes against renormalised degree-preserving nulls; `ploscb/p3b_block_precise.py`: block model of all four connectomes with 1,024-probe effective ranks.
4. `ploscb/p4_sex_individual.py`: three brains and three nerve cords on shared cell types (relative capture, difference spectra); `ploscb/p4b_similarity.py`: entrywise and reliability-corrected similarity; `ploscb/p4c_sexmode.py`: the leading male–female difference mode.
5. `ploscb/p5_patches.py <case>` for `male_brain male_cns female_brain female_cns`: sex-specific neurons against 50 matched random draws (removal, containment and patch rank).
6. `ploscb/p6_robustness.py`: thresholds of 1–10 synapses, binary and log weights, and validation of the Hutchinson and randomized-SVD estimators against exact dense SVD.
7. `ploscb/p8_scaling_modes.py`: rank scaling with network size and mode localisation. Localisation needs the leading singular vectors (`results/modes_*_count.npz` from `spectrum_main.py`, `ploscb/results/modes_*_count.npz` from `p2_spectra.py`), which are not included because they exceed GitHub's file-size limit; without them the script computes the scaling only.
8. `ploscb/p9_precise_pr.py`: 1,024-probe effective ranks with standard errors for all 15 networks and both weightings (`ploscb/results/pr_table.json`).
9. `ploscb/fig1_rank.py` … `ploscb/fig5_patches.py`, `ploscb/figsupp_plos.py`: Figs 1–5 and S1–S3 Figs (PLOS format: 190 mm wide, Arial 8 pt, PDF, PNG and LZW-compressed TIFF at 450 dpi; `figlib_plos.py` holds the shared styling). `ploscb/make_tables.py`: S1 Table and Table 1.

`speclib.py` implements the exact Frobenius energy, the Hutchinson estimate of Σσ⁴ (participation ratio), randomized SVD, input normalisation and neurotransmitter signing, and the three null models. The optional panel-alignment gate of the figure scripts runs only if the `audit_panel_alignment` module is on the path (`$NATURE_FIGURE_SCRIPTS`).

## Citation

Please cite the paper (reference to be added on publication) and the Zenodo archive of the release you used (v2.0.0: https://doi.org/10.5281/zenodo.23088621; v1.0.0: https://doi.org/10.5281/zenodo.22760814; all versions: https://doi.org/10.5281/zenodo.22760813), together with the connectome papers and data deposits: MaleCNS (Berg et al., 2026), FlyWire (Dorkenwald et al., 2024; Schlegel et al., 2024), BANC (Bates et al., 2026, and the Dataverse deposit https://doi.org/10.7910/DVN/7WTH1N) and MANC (Takemura et al., 2024; Marin et al., 2024).

## License

Code: MIT License. Derived matrices, results and figures: CC BY 4.0 (derived from source data released under CC BY 4.0).
