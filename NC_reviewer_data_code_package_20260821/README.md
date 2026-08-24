# Reviewer data and code package

Manuscript: *Multiscale detection of cross-pathogen epidemic transitions under Pathogen X-like surveillance constraints*

This archive is intentionally minimal. It contains the custom code central to the claims, immutable locked configurations, the inputs and cached structural scores needed to rerun the independently seeded confirmatory benchmark, its expected outputs, and numerical source data underlying the data-bearing main and Supplementary figures/tables. It excludes exploratory notebooks, superseded runs, plotting-only scripts, intermediate tuning grids, COVID-19 analyses not reported in this manuscript, and manuscript-production utilities.

## Contents

- `code/epidemic_phase_segmentation_v31_1_archetype_adaptive.py`: locked EEMD–CWT phase-segmentation implementation.
- `code/cross_pathogen_phase_extension_v1.py`: thin locked cross-pathogen/real-world application layer.
- `code/evaluate_sensitivity_prioritized_confirmation.py`: one-command entry point for the independently seeded confirmatory benchmark.
- `code/unified_operational_benchmark.py` and the `optimize_*`/`evaluate_*` modules: dependencies of that confirmatory entry point.
- `config/`: locked phase-segmentation parameters, pre-confirmatory operating lock and original environment record.
- `data/`: confirmatory inputs, cached locked EEMD–CWT scores and development quantities required by the entry point.
- `expected_outputs/confirmatory/`: frozen outputs reported in the manuscript and Supplementary Information.
- `source_data/`: numerical values underlying Figs. 2–5, Tables 3–5 and Supplementary Tables 4 and 6–9. Figure 4 and Supplementary Tables 6–8 use `expected_outputs/confirmatory/`.
- `package_manifest_sha256.csv`: file sizes and SHA-256 checksums for integrity checking.

Figure 1 and main Tables 1–2 are conceptual/definitional and therefore have no numerical source-data file. Supplementary Tables 1–3 and 5 are protocol/specification tables represented by the manuscript, locked JSON configurations and this package documentation.

## Environment

Python 3.10 or later is recommended. From the package root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The scientific dependencies are NumPy, pandas, SciPy and Matplotlib. The confirmatory rerun uses the cached, locked EEMD–CWT structural-score matrix so that reviewers do not need to repeat the computationally expensive feature extraction before evaluating the prespecified operating point. The raw independently seeded integer-count arrays and their metadata are nevertheless included.

## Reproduce the confirmatory benchmark

Run from the package root:

```bash
python code/evaluate_sensitivity_prioritized_confirmation.py
```

The command creates `reproduced_outputs/confirmatory/` and refuses to overwrite an existing result directory. Compare the generated CSV/JSON files with `expected_outputs/confirmatory/`.

The primary expected results are:

- confirmatory set: 180 outbreak and 180 paired no-transition trajectories across six operational profiles;
- EEMD–CWT/Rt concordance sensitivity: 155/180 (86.1%) in the primary observable-expansion window;
- achieved alert burden: 2.320 alert-days per 100 no-transition surveillance days;
- day-level specificity: 97.68%;
- strict earlier-boundary sensitivity: 118/180 (65.6%).

## Source-data map

| Manuscript item | Package location |
|---|---|
| Fig. 2 | `source_data/Figure_2/` |
| Fig. 3 and Table 3 | `source_data/Figure_3_and_Table_3/` |
| Fig. 4 | `expected_outputs/confirmatory/locked_endpoint_summary.csv` and `paired_primary_vs_comparators.csv` |
| Fig. 5 and Table 4 | `source_data/Figure_5_and_Table_4/` |
| Table 5 | `source_data/Table_5/` |
| Supplementary Table 4 | `source_data/Supplementary_Table_4/` |
| Supplementary Tables 6–8 | `expected_outputs/confirmatory/` |
| Supplementary Table 9 | `source_data/Supplementary_Table_9/` |

## Data scope and ethics

No individual-level participant data are included or analysed. Synthetic datasets contain simulated integer-count surveillance trajectories. Real-world files are processed aggregate surveillance counts and derived landmarks. The independent national reference protocol and weekly evidence table were constructed from publicly available China National Influenza Center reports.

## Reviewer access and release

Upload this entire archive as a private peer-review capsule/repository and enable anonymous reviewer access. Do not place author names in the public-facing repository description during double-anonymous review. Replace the placeholder in the manuscript only after the private reviewer URL has been generated. A DOI-minting public release can be made upon acceptance.
