"""Preprocessing: raw datasets -> the 8 normalized Hazard factors (Section 2.3, Eqs. 2-9).

Each factor is built in two steps, as in the original analysis:

1. normalization to [0, 1] on a working grid of about 30 m
   (geographic grid for SAI, BUI, LSI and FFI; UTM 36N grid for CPI, RNI, FHI and SLRI);
2. :func:`align` - bilinear resampling onto the common 30 m grid in EPSG:32636,
   cells without a value inside the island set to 0, land mask applied.

The vector inputs (boundary, coastline polygon, roads, rivers, burned areas) were
stored in PostGIS: export the PostGIS layers to the GeoPackage files listed under
inputs (config/default.yaml), so that this module runs without a database.

Note: the 8 aligned factors used in the article are archived as underlying data.
They are the authoritative inputs of scripts 01-06.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import Affine, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import distance_transform_edt

log = logging.getLogger(__name__)
GEO = "EPSG:4326"


# =================================================================== grids
def boundary(cfg, crs: str) -> gpd.GeoDataFrame:
    """Cyprus land boundary in ``crs``."""
    return gpd.read_file(cfg.input("boundary")).to_crs(crs)


def geographic_grid(bounds, res_m: float, pad_deg: float = 0.0):
    """Grid in degrees of about ``res_m`` metres (metres per degree at the mean latitude)."""
    minx, miny, maxx, maxy = bounds[0] - pad_deg, bounds[1] - pad_deg, bounds[2] + pad_deg, bounds[3] + pad_deg
    m_per_deg = 111320 * np.cos(np.radians((miny + maxy) / 2))
    width = int((maxx - minx) * m_per_deg / res_m)
    height = int((maxy - miny) * m_per_deg / res_m)
    return from_bounds(minx, miny, maxx, maxy, width, height), width, height


def utm_grid(bounds, res_m: float):
    """North-up grid of ``res_m`` metres starting at the upper-left corner of ``bounds``."""
    minx, miny, maxx, maxy = bounds
    width, height = int((maxx - minx) / res_m), int((maxy - miny) / res_m)
    return Affine.translation(minx, maxy) * Affine.scale(res_m, -res_m), width, height


def reference_grid(cfg) -> dict:
    """Common 30 m grid of the factors: extent of the island in UTM 36N."""
    res = cfg["preprocess"]["resolution_m"]
    bounds = boundary(cfg, cfg.crs).total_bounds
    width, height = int((bounds[2] - bounds[0]) / res), int((bounds[3] - bounds[1]) / res)
    return {"crs": cfg.crs, "transform": from_bounds(*bounds, width, height), "width": width, "height": height}


def land_mask(geoms, transform, shape) -> np.ndarray:
    """True on land (inside the boundary)."""
    return ~geometry_mask(list(geoms), out_shape=shape, transform=transform, invert=False)


def _profile(crs, transform, width, height, nodata=np.nan) -> dict:
    return {"driver": "GTiff", "dtype": "float32", "count": 1, "crs": crs, "transform": transform,
            "width": width, "height": height, "nodata": nodata, "compress": "deflate"}


def _warp(src, transform, width, height, resampling) -> np.ndarray:
    """Read ``src`` (same CRS) resampled onto another grid, NaN outside."""
    with WarpedVRT(src, width=width, height=height, transform=transform, resampling=resampling,
                   nodata=np.nan) as vrt:
        return vrt.read(1)


# ======================================================== continuous factors
def seismic(cfg) -> tuple[np.ndarray, dict]:
    """SAI: GEM PGA (475-year return period), min-max over Cyprus (Eq. 2).

    Bilinear resampling of the 0.05 degree (about 5 km) raster to the ~30 m grid.
    Reconstructed from the Methods: the original script is not available.
    """
    res = cfg["preprocess"]["resolution_m"]
    land = boundary(cfg, GEO)
    transform, w, h = geographic_grid(land.total_bounds, res, pad_deg=0.1)
    with rasterio.open(cfg.input("gem_pga")) as src:
        pga = _warp(src, transform, w, h, Resampling.bilinear).astype(np.float32)
    pga[~land_mask(land.geometry, transform, pga.shape)] = np.nan
    sai = (pga - np.nanmin(pga)) / (np.nanmax(pga) - np.nanmin(pga))
    return sai, _profile(GEO, transform, w, h)


def built_up(cfg) -> tuple[np.ndarray, dict]:
    """BUI: GHS-BUILT-S 2018 (10 m, 0-100 m2 per cell) / 100 (Eq. 3), average to ~30 m."""
    res = cfg["preprocess"]["resolution_m"]
    land = boundary(cfg, GEO)
    transform, w, h = geographic_grid(land.total_bounds, res, pad_deg=0.1)
    with rasterio.open(cfg.input("ghs_built_2018")) as src:
        data = np.clip(src.read(1).astype(np.float32) / 100.0, 0, 1)
        prof = src.profile.copy()
    prof.update(dtype="float32", nodata=np.nan)
    with rasterio.io.MemoryFile() as mem, mem.open(**prof) as tmp:
        tmp.write(data, 1)
        bui = _warp(tmp, transform, w, h, Resampling.average)
    bui[~land_mask(land.geometry, transform, bui.shape)] = np.nan
    return bui, _profile(GEO, transform, w, h)


def landslide(cfg) -> tuple[np.ndarray, dict]:
    """LSI: ELSUS v2, 5 ordinal classes -> 0, 0.25, 0.5, 0.75, 1 (Eq. 4).

    The original script resamples the 200 m classes to ~30 m with bilinear
    interpolation (the Methods text says nearest neighbour).
    """
    res = cfg["preprocess"]["resolution_m"]
    with rasterio.open(cfg.input("elsus")) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        lo, hi = np.nanmin(data), np.nanmax(data)
        lsi = np.clip((data - lo) / (hi - lo), 0, 1)
        prof = src.profile.copy()
        scale = 200 / res
        w, h = int(src.width * scale), int(src.height * scale)
        transform = from_bounds(*src.bounds, w, h)
    prof.update(dtype="float32", nodata=np.nan)
    with rasterio.io.MemoryFile() as mem, mem.open(**prof) as tmp:
        tmp.write(lsi, 1)
        out = _warp(tmp, transform, w, h, Resampling.bilinear)
    return out, _profile(prof["crs"], transform, w, h)


def soil_loss(cfg) -> tuple[np.ndarray, dict]:
    """SLRI: RUSLE soil loss (t/ha/yr) / 10, capped at 1 (Eq. 9); 100 m -> 30 m bilinear."""
    p = cfg["preprocess"]
    with rasterio.open(cfg.input("rusle")) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        slri = np.clip(data / p["rusle_cap"], 0, 1)
        land = boundary(cfg, src.crs)
        transform, w, h = calculate_default_transform(src.crs, src.crs, src.width, src.height,
                                                      *land.total_bounds, resolution=p["resolution_m"])
        out = np.full((h, w), np.nan, dtype=np.float32)
        reproject(slri, out, src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                  dst_crs=src.crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    out[~land_mask(land.geometry, transform, out.shape)] = np.nan
    return out, _profile(src.crs, transform, w, h)


# ============================================================ vector factors
def _vector_grid(cfg):
    land = boundary(cfg, cfg.crs)
    transform, w, h = utm_grid(land.total_bounds, cfg["preprocess"]["resolution_m"])
    return land, transform, w, h


def proximity(cfg, layer_key: str, buffer_m: float) -> tuple[np.ndarray, dict]:
    """RNI (roads, 250 m) and FHI (watercourses, 150 m): 1 within the buffer, 0 outside (Eqs. 7-8)."""
    land, transform, w, h = _vector_grid(cfg)
    lines = gpd.read_file(cfg.input(layer_key)).to_crs(cfg.crs)
    zone = lines.geometry.buffer(buffer_m).union_all()
    r = rasterize([(zone, 1)], out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8, all_touched=True)
    r[~land_mask(land.geometry, transform, r.shape)] = 0
    return r.astype(np.float32), _profile(cfg.crs, transform, w, h, nodata=0)


def coastal_proximity(cfg) -> tuple[np.ndarray, dict]:
    """CPI: distance bands to the shoreline, 1 / 0.67 / 0.33 / 0 (Eq. 5).

    The distance is the Euclidean distance of each land cell to the nearest
    cell outside the land polygon of the coastline layer.
    """
    p = cfg["preprocess"]
    land, transform, w, h = _vector_grid(cfg)
    coast = gpd.read_file(cfg.input("coastline_polygon")).to_crs(cfg.crs)
    inside = rasterize([(coast.geometry.union_all(), 1)], out_shape=(h, w), transform=transform,
                       fill=0, dtype=np.uint8, all_touched=True)
    distance = distance_transform_edt(inside > 0) * p["resolution_m"]
    cpi = np.select([distance < b for b in p["cpi_bands_m"]], p["cpi_values"][:-1], p["cpi_values"][-1])
    cpi = cpi.astype(np.float32)
    cpi[~land_mask(land.geometry, transform, cpi.shape)] = 0
    return cpi, _profile(cfg.crs, transform, w, h, nodata=0)


def fire_footprint(cfg) -> tuple[np.ndarray, dict]:
    """FFI: burned areas (EFFIS, buffered 100 m), 0.25 + 0.75 exp(-0.5 t) per fire,
    summed over overlapping fires and capped at 1 (Eq. 6). t = reference year - fire year."""
    p = cfg["preprocess"]
    land = boundary(cfg, GEO)
    transform, w, h = geographic_grid(land.total_bounds, p["resolution_m"])
    fires = gpd.read_file(cfg.input("effis_fires")).to_crs(GEO)
    years = pd.to_datetime(fires["finaldate"]).dt.year
    t = p["fire_reference_year"] - years
    value = np.where(t >= 0, p["fire_floor"] + (1 - p["fire_floor"]) * np.exp(-p["fire_decay_k"] * t), 0.0)
    ffi = np.zeros((h, w), dtype=np.float32)
    for geom, v in zip(fires.geometry, value):
        ffi += rasterize([(geom, float(v))], out_shape=(h, w), transform=transform, fill=0, dtype=np.float32)
    ffi = np.clip(ffi, 0, 1)
    ffi[~land_mask(land.geometry, transform, ffi.shape)] = 0
    return ffi, _profile(GEO, transform, w, h, nodata=0)


# ================================================================ alignment
def align(array: np.ndarray, profile: dict, grid: dict, mask: np.ndarray) -> np.ndarray:
    """Bilinear resampling onto the common grid; empty cells on land -> 0; NaN off land."""
    src = np.where(np.isfinite(array), array, np.nan).astype(np.float32)
    if profile.get("nodata") is not None and np.isfinite(profile["nodata"]):
        src[array == profile["nodata"]] = np.nan
    out = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
    reproject(src, out, src_transform=profile["transform"], src_crs=profile["crs"],
              dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    out = np.where(np.isnan(out), 0, out).astype(np.float32)
    out[~mask] = np.nan
    return out


BUILDERS = {
    "SAI": seismic,
    "BUI": built_up,
    "LSI": landslide,
    "CPI": coastal_proximity,
    "FFI": fire_footprint,
    "RNI": lambda cfg: proximity(cfg, "roads", cfg["preprocess"]["road_buffer_m"]),
    "SLRI": soil_loss,
    "FHI": lambda cfg: proximity(cfg, "rivers", cfg["preprocess"]["river_buffer_m"]),
}


def build_factor(cfg, code: str, grid: dict, mask: np.ndarray, out_path: Path) -> None:
    """Build one factor and write it on the common grid."""
    array, profile = BUILDERS[code](cfg)
    aligned = align(array, profile, grid, mask)
    prof = _profile(grid["crs"], grid["transform"], grid["width"], grid["height"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(aligned, 1)
    log.info("%s written (%d land pixels)", code, int(np.isfinite(aligned).sum()))
