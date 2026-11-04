# Data

## In this repository: `data/sites/`

| File | Content | Source / licence |
|---|---|---|
| `site_points.geojson` | 34 site points: `id`, `name`, `district` (WGS 84, EPSG:4326) | derived from OpenStreetMap, (c) OpenStreetMap contributors, ODbL |
| `site_polygons.geojson` | site footprints, matched to the points (nearest polygon within 300 m) | OpenStreetMap, ODbL |
| `site_attributes.csv` | Heritage status, Enclosure, Shelter | UNESCO World Heritage List; photo-interpretation of Google Maps / Google Earth Pro imagery (2026) |
| `site_raster_values.csv` | Land Use (WorldCover code), elevation (m), slope (deg), distance to coast (m) at each point | output of `scripts/02_vulnerability_index.py --sample-rasters` |
| `exposure_reviews.csv` | Google Places review counts, July 2026 | Google Places API; see the note below |

Licence of the tables produced in this work: CC BY 4.0. Data derived from
OpenStreetMap keep the ODbL licence and require the attribution above.

**Google Places.** The review counts are an aggregate (one number per site)
collected in July 2026. The Google Maps Platform terms restrict storing and
redistributing Places content: check that sharing these aggregates is allowed
for your use.

## Archived separately (Zenodo): `data/factors/`

The 8 normalized Hazard factors on the common grid (30 m, EPSG:32636, float32,
NaN outside the island), about 130 MB in total:

`SAI.tif`, `BUI.tif`, `LSI.tif`, `CPI.tif`, `FFI.tif`, `RNI.tif`, `SLRI.tif`, `FHI.tif`

These are the exact inputs of the article. Download them from the Zenodo record
of the article (DOI to be added) into `data/factors/`.

## Raw datasets (not redistributed): `data/raw/`

Needed only to rebuild the factors (`scripts/00_preprocess_factors.py`) or the
raster values of the sites. Download them from the providers:

| Factor / attribute | Dataset | Provider |
|---|---|---|
| SAI | GEM Global Seismic Hazard Map v2023.1, PGA 475 years, rock | Global Earthquake Model Foundation |
| BUI | GHS-BUILT-S R2023A, epoch 2018, 10 m | European Commission, JRC |
| LSI | ELSUS v2 (European Landslide Susceptibility Map) | European Commission, JRC / ESDAC |
| SLRI | RUSLE2015 soil loss by water | ESDAC |
| CPI, RNI, FHI, Distance to Coast | EuroRegionalMap 2025 | EuroGeographics |
| FFI | EFFIS burned areas 2003-2018 | European Forest Fire Information System |
| Land Use | ESA WorldCover 10 m 2021 v200 | ESA |
| Elevation, Slope | SRTM Global 1 arc-second v3 | NASA JPL / LP DAAC |

Expected file names and folders are listed under `inputs` in `config/default.yaml`.
The vector layers (boundary, coastline polygon, roads, rivers, burned areas) were
stored in PostGIS: export the PostGIS layers to the GeoPackage files listed under
inputs.
