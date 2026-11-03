# Verification against the article

The code of this repository was rebuilt from the scripts used for the article
(Open Research Europe, Draft v3, 24 October 2026). Every number below was
recomputed with this code and compared with the original outputs and with the
text of the article. `tests/test_reproduce_article.py` repeats the comparison
automatically.

## 1. Results reproduced exactly

| Element of the article | Check | Result |
|---|---|---|
| Hazard Index raster (Figure 4) | pixel by pixel vs the raster of the article | identical (max difference 0.0, same 10,285,871 pixels) |
| PCA pre-screening (Section 3.1) | eigenvalues, explained variance, loadings | identical (3 PCs Kaiser, 6 retained, 84.4 %) |
| Cascade weights (Table 5) | 8 weights | identical: SAI 0.131, BUI 0.132, LSI 0.262, CPI 0.101, FFI 0.049, RNI 0.093, SLRI 0.183, FHI 0.049 |
| Vulnerability Index (Table 6) | raw values, 7 scores, VI, classes, thresholds | identical (classes 10/7/17; thresholds 83.6 / 270.0 m, 7,131 / 21,980 m, 3.0 / 5.4 deg) |
| Exposure Index | E_norm, classes | identical (classes 3/10/10/8/3) |
| Risk Index (Table 7) | HI at the sites, all classes, R | identical for the 34 sites (classes 17/12/5, mean 28.3, median 23.5) |
| OAT pixel level (Table 8) | all columns | identical, see 3.4 |
| OAT stability (Figure 7) | shares of pixels | identical: 76.5 / 13.3 / 7.1 / 3.1 / < 0.05 % |
| OAT sites (Figure 8) | per site counts and rank ranges | identical to `OAT_sites_per_site.csv` |
| Without Enclosure and Shelter | VI and Risk classes | 16 and 7 changes, 5 High sites kept |
| Site polygon support | VI and Risk classes | 5 and 4 changes, Kourion and Agios Georgios added |
| Fixed thresholds | Risk classes | 3 / 4 / 7 changes, Spearman of Risk 0.82-0.90 |
| Coastal inputs | weights without CPI, classes | identical (0.163, 0.146, 0.275, 0.054, 0.104, 0.207, 0.051; 8 / 10 / 13 changes) |
| Product of continuous indices | Spearman, top five | 0.77, 4 shared, Kourion replaces Marion Ruins |
| Figures 2-9 | image vs the published PNG | same size; 89-99 % identical pixels, the rest from fonts (Linux vs Windows); Figure 9 checked visually: identical content |

## 2. Steps rebuilt because the original script was missing

* **Alignment of the factors** (`preprocess.align`). The script that produced the
  `*_aligned.tif` files was not in the code folder. The rule was rebuilt and
  tested on the stored intermediate rasters: bilinear resampling onto the
  common grid, empty land cells set to 0, land mask. Result identical for SAI,
  BUI, CPI, FFI, RNI, SLRI and FHI (max difference 3e-8), LSI within 3e-4.
* **SAI** (`preprocess.seismic`). `Tectonic.py` contained a copy of the BUI
  script. SAI was rebuilt from the Methods (GEM PGA, min-max over Cyprus,
  bilinear resampling).
* The land boundary and four vector layers (coastline polygon, roads,
  watercourses, burned areas) were read from PostGIS. The file
  `Data/Boundaries/Cyprus.shp` differs slightly from the PostGIS boundary
  (252 pixels of the land mask), so the preprocessing step reproduces the
  archived factors exactly only with the layers exported from PostGIS
  (`scripts/export_postgis_layers.py`). The 8 archived factors are therefore
  the authoritative input of the article.

## 3. Differences between the text of the article and the code

To be corrected in the text (the code gives the published results):

1. **Pixel sample of the correlation and PCA** (Section 2.3, "Pre-screening"):
   the text says a random 15 % sample; the results of the article (eigenvalues,
   correlations, weights) come from a 2 % sample with seed 42.
2. **Mean rank aggregation** (Section 3.5): the text says that the mean rank
   shares 4 of the 5 highest sites and that Choirokoitia replaces Marion Ruins.
   The code gives 3 shared sites: the five highest are Amathus, Chrysopolitissa
   complex, Lempa-Lakkoi, The Necropolis at Meletis Forest and Agios Georgios
   Ancient Basilicas (the last two tied at 22.83); Old Catacombs (22.50) and
   Marion Ruins (21.50) are out, Choirokoitia is sixth (22.67).
3. **Fixed thresholds** (Section 3.5): the Spearman correlation of the
   Vulnerability Index with the original is 0.90-0.92 (text: 0.88-0.92).
4. **Table 8, FHI**: mean normalized value 0.127 (text: 0.128).
5. **LSI resampling** (Section 2.3): the text says nearest neighbour and
   LSI in {0, 0.25, 0.5, 0.75, 1}; the code resamples with bilinear
   interpolation, so LSI is continuous between the class values.
6. **CPI and FFI** (Section 2.3): the text says they are rasterized directly at
   30 m with no resampling; in the code they are rasterized on an intermediate
   grid and then resampled (bilinear) onto the common grid, so CPI has
   intermediate values at the band limits. RNI and FHI stay binary.
   CPI bands use d < 200 m (text: d <= 200 m).
7. **SLRI source file**: the code reads `Erosion_2016_UTM_Cyprus.tif`; the
   text describes RUSLE2015 with inputs of 2010. Check that the file is the
   RUSLE2015 product.
8. **Table 8 and Figure 7 use two versions of the OAT**: Table 8 perturbs the
   weighted sum of the normalized factors (Eq. 24); Figure 7 and the site-level
   OAT perturb the propagated factors and rescale the Hazard Index to [0, 1],
   as in the main analysis. Both are kept as published; the Methods could say
   which surface each result uses.
9. **Exposure query**: the review counts of July 2026 were collected on the
   polygons of `Arch_sites_v3.shp`; `scripts/collect_google_reviews.py` uses
   `data/sites/site_polygons.geojson`, so a new run is not strictly the same query.

## 4. How to repeat the verification

```bash
python scripts/01_hazard_index.py
python scripts/02_vulnerability_index.py
python scripts/03_exposure_index.py
python scripts/04_risk_index.py
python scripts/05_sensitivity.py
pytest -q
```
