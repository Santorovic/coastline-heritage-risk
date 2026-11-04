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
| **Risk (R)** | combination of the three | R = H class x V class x E class |

Every class is computed from the data with Multi-Otsu thresholding, which places the class limits where the values separate best, with no threshold set by hand.

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
data/sites/                  the 34 sites: points, polygons, attributes, review counts
tests/                       unit tests + regression tests against the article
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

Notes for Windows with conda:

- `pip install -e .` is required before running the scripts: it installs the
  `coastline` package that every script imports. (Alternatively, set
  `PYTHONPATH=src`.)
- Run the scripts from an **activated** environment (`conda activate <env>`). If you
  call the environment's `python.exe` directly, without activating it, its
  `Library\bin` folder is not on `PATH` and script 01 can crash silently inside the
  PCA (scikit-learn, Windows error code `0xc06d007f`, no Python traceback). The fix is
  to activate the environment, or to put `<env>\Library\bin`, `<env>\Library\mingw-w64\bin`
  and `<env>\Scripts` on `PATH`.

## Data

`data/sites/` (in this repository) holds the site points and polygons, the
attributes scored by photo-interpretation, the values read from the rasters and
the review counts. The 8 normalized Hazard factors (about 130 MB) are archived
on Zenodo: download them into `data/factors/` (see [data/README.md](data/README.md)).

To use data stored elsewhere, create `config/local.yaml` (ignored by Git) with only
the values that differ on your computer, or set `COASTLINE_DATA_DIR`.

### Land Use input: ESA WorldCover v200

The Land Use attribute of the Vulnerability Index comes from **ESA WorldCover 10 m
2021 v200** (reference year 2021; Zanaga et al., 2022, https://doi.org/10.5281/zenodo.7254221).
Cyprus is covered by two 3-degree tiles:

- `ESA_WorldCover_10m_2021_v200_N33E030_Map.tif` (30-33 E, 33-36 N)
- `ESA_WorldCover_10m_2021_v200_N33E033_Map.tif` (33-36 E, 33-36 N)

The Cyprus mosaic `ESA_WorldCover_10m_2021_v200_Cyprus.tif` (the file named under
`inputs.worldcover`) is the plain union of the two tiles, with no resampling and no
reclassification (same CRS EPSG:4326, same 10 m grid, uint8, nodata 0):

```python
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge

tiles = ["ESA_WorldCover_10m_2021_v200_N33E030_Map.tif",
         "ESA_WorldCover_10m_2021_v200_N33E033_Map.tif"]
merge(tiles, method="first", nodata=0, dtype="uint8", resampling=Resampling.nearest,
      dst_path="ESA_WorldCover_10m_2021_v200_Cyprus.tif",
      dst_kwds={"compress": "deflate", "tiled": True, "blockxsize": 1024,
                "blockysize": 1024, "BIGTIFF": "IF_SAFER"})
```

These rasters are **not distributed in this repository** (download the tiles from
ESA WorldCover, then point `inputs.worldcover` to the mosaic in `config/local.yaml`).
In WorldCover v200 the sea is class 80 (permanent water), not nodata. The Land Use
column of `data/sites/site_raster_values.csv` holds the value of this mosaic at each
site point; `scripts/02_vulnerability_index.py --sample-rasters` reads it again
(it writes `results/vulnerability/site_raster_values.csv`).

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
rasters are not available. The maps written by script 06 are previews.

Script `00_preprocess_factors.py` is not part of this sequence: it rebuilds the 8
normalized factors from the raw datasets listed under `inputs` in
`config/default.yaml`, which are **not distributed**. It writes to the paths of
`inputs.factors`, so if those point to existing factor files they are overwritten.
To reproduce the article start from script 01 with the archived factors.

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
| Descriptive statistics of the Results section | `scripts/07_descriptive_statistics.py` | `results/descriptive/statistics.csv` |

## Citation

See [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Funding

This work was supported by the European Union's Horizon Europe research and
innovation programme under the Marie Sklodowska-Curie grant agreement
No 101130576 (COASTLINE).

## License

Code: MIT (see [LICENSE](LICENSE)). Data in `data/sites/`: see [data/README.md](data/README.md).
