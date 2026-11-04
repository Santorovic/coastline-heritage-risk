# COASTLINE - multi-hazard risk of coastal archaeological heritage

Code of the article

> **A data-driven multi-hazard risk framework based on open geospatial datasets for
> coastal heritage: a case study of 34 sites in Cyprus.** Open Research Europe (under review).

The framework computes four indices for each archaeological site:

| Index | What it measures | Method |
|---|---|---|
| **Hazard (H)** | 8 natural and anthropogenic factors, mapped over the whole island, that drive the physical degradation of archaeological remains | PCA pre-screening + cascade weighting (network centrality) |
| **Vulnerability (V)** | intrinsic physical condition of each site, from 7 attributes | scores 1-3, equal-weight mean |
| **Exposure (E)** | presence of visitors, as a proxy of the value at stake | Google Places review counts, log scale |
| **Risk (R)** | combination of the three | R = H class x V class (1, 3, 5) x E class |

Every class is computed from the data with Multi-Otsu thresholding, which allows to build naturalbreaks with a data-driven approach. 

## Repository structure

```
config/default.yaml          every path and parameter of the article (one place)
src/coastline/               the package: 
    config.py                reads the configuration
    hazard.py                PCA pre-screening, cascade weighting, Hazard Index
    vulnerability.py         raster sampling, scores, Vulnerability Index
    exposure.py              Exposure Index
    risk.py                  Risk Index
    classify.py              Multi-Otsu classification
    preprocess.py            raw datasets -> 8 normalized factors
    sensitivity/             sensitivity analyses 
    figures/paper.py         figures of the article
scripts/                     the steps, run in order (each one is short)
    00_preprocess_factors.py   optional: rebuild the factors from raw data
    01_hazard_index.py ... 06_figures.py
    collect_google_reviews.py  optional: collect new review counts
    export_postgis_layers.py   optional: export the PostGIS layers once
data/sites/                  the 34 sites: points, polygons, attributes, review counts
tests/                       unit tests + regression tests against the article
docs/                        verification report, map from the article to the code
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/<github-user>/coastline-heritage-risk.git
cd coastline-heritage-risk
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows, if `rasterio` or `geopandas` do not install with pip, use conda:
`conda install -c conda-forge geopandas rasterio scikit-image scikit-learn`.

## Data

`data/sites/` (in this repository) holds the site points and polygons, the
attributes scored by photo-interpretation, the values read from the rasters and
the review counts. The 8 normalized Hazard factors (about 130 MB) are archived
on Zenodo: download them into `data/factors/` (see [data/README.md](data/README.md)).

To use data stored elsewhere, copy `config/local.example.yaml` to
`config/local.yaml` and write your paths there, or set `COASTLINE_DATA_DIR`.

## Run

```bash
python scripts/01_hazard_index.py         # Hazard Index raster, weights, PCA      (~15 s)
python scripts/02_vulnerability_index.py  # Vulnerability Index                     (~1 s)
python scripts/03_exposure_index.py       # Exposure Index                          (~1 s)
python scripts/04_risk_index.py           # Risk Index of the 34 sites              (~5 s)
python scripts/05_sensitivity.py          # all sensitivity analyses               (~4 min)
python scripts/06_figures.py              # figures                                (~1 min)
pytest -q                                 # checks that the results equal the article
```

Outputs are written to `results/`. Script 05 accepts `--parts` to run one analysis
(e.g. `--parts pixels sites`) and `--skip-polygon` if the DEM, slope and WorldCover
rasters are not available. Maps of the article are finished in ArcGIS Pro: script 06
writes previews and, in `results/figures/arcgis/`, the layers with their symbology.

## From the article to the code

| Article | Code | Output |
|---|---|---|
| Eq. 1, Multi-Otsu | `classify.multiotsu_classes` | - |
| Eqs. 2-9, normalization | `preprocess.py` | `data/factors/*.tif` |
| Eqs. 10-12, pre-screening | `hazard.redundancy_screening` | `results/hazard/pca_*.csv` |
| Eqs. 13-16, cascade weighting, Table 5 | `hazard.cascade_weighting` | `results/hazard/weights.csv` |
| Eq. 17, Hazard Index, Figure 4 | `hazard.hazard_index` | `results/hazard/hazard_index.tif` |
| Eqs. 18-19, Table 4, Table 6 | `vulnerability.py` | `results/vulnerability/` |
| Eq. 20 | `exposure.py` | `results/exposure/` |
| Eq. 21, Table 7, Figures 5-6 | `risk.py`, `io.site_buffer_max` | `results/risk/risk_index.csv` |
| Eqs. 22-28, Table 8, Figures 7-8 | `sensitivity/weights.py` | `results/sensitivity/oat_*` |
| Vulnerability tests, Figure 9 | `sensitivity/vulnerability.py` | `results/sensitivity/vulnerability_tests*.csv` |
| Aggregation of the Risk Index | `sensitivity/aggregation.py` | `results/sensitivity/aggregation*.csv` |
| Coastal inputs | `sensitivity/coastal.py` | `results/sensitivity/coastal_tests*.csv` |

[docs/VERIFICATION.md](docs/VERIFICATION.md) lists the checks against the
article and the points where the text and the code differ.

## Citation

See [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Funding

This work was supported by the European Union's Horizon Europe research and
innovation programme under the Marie Sklodowska-Curie grant agreement
No 101130576 (COASTLINE).

## License

Code: MIT (see [LICENSE](LICENSE)). Data in `data/sites/`: see [data/README.md](data/README.md).
