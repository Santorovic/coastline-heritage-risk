"""Reading and writing rasters, sites and tables."""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio.io import MemoryFile

from .config import Config

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ rasters
def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Band 1 as float32 with NaN where the raster has no data, and its profile."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
        return arr, src.profile.copy()


def write_raster(path: Path, array: np.ndarray, profile: dict, nodata: float = -9999.0) -> None:
    """Write a float32 GeoTIFF; NaN cells are written as ``nodata``."""
    prof = profile.copy()
    prof.update(driver="GTiff", dtype="float32", count=1, nodata=nodata, compress="deflate")
    out = np.where(np.isfinite(array), array, nodata).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)
    log.info("written %s", path)


def load_factors(cfg: Config, codes: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Load the normalized Hazard factors (all on the same grid).

    Returns a dict ``code -> array`` and the raster profile of the grid.
    """
    codes = codes or cfg.factor_codes
    arrays, profile = {}, None
    for code in codes:
        arr, prof = read_raster(cfg.input("factors", code))
        if profile is None:
            profile = prof
        elif (prof["width"], prof["height"], prof["transform"]) != (
            profile["width"], profile["height"], profile["transform"]
        ):
            raise ValueError(f"{code} is not on the same grid as {codes[0]}")
        arrays[code] = arr
    log.info("loaded %d factors (%d x %d pixels)", len(codes), profile["width"], profile["height"])
    return arrays, profile


def valid_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels where every factor has a value (the valid land pixels)."""
    mask = None
    for arr in arrays.values():
        mask = np.isfinite(arr) if mask is None else mask & np.isfinite(arr)
    return mask


# -------------------------------------------------------------------- sites
def read_sites(cfg: Config, kind: str = "points") -> gpd.GeoDataFrame:
    """The sites (``id``, ``name``, geometry) in the working CRS, sorted by id.

    ``kind`` is "points" (the 34 site points) or "polygons" (site footprints).
    """
    gdf = gpd.read_file(cfg.input(f"site_{kind}")).to_crs(cfg.crs)
    if not gdf["id"].is_unique:
        raise ValueError("duplicate site id in the sites file")
    return gdf.sort_values("id").reset_index(drop=True)


def read_table(path: Path) -> pd.DataFrame:
    """CSV table sorted by site id."""
    return pd.read_csv(path).sort_values("id").reset_index(drop=True)


# ------------------------------------------------- Hazard Index at the sites
def _buffer_values(src, point, radius_m: float, radius_max_m: float):
    """Valid raster values in a buffer around ``point``; the radius is doubled
    until at least one valid pixel is found (up to ``radius_max_m``).

    Returns (values, radius used). ``all_touched=True``: every pixel touched by
    the buffer counts, as in the original analysis.
    """
    radius = radius_m
    while radius <= radius_max_m:
        try:
            out, _ = rasterio.mask.mask(src, [point.buffer(radius)], crop=True,
                                        all_touched=True, filled=False)
            vals = np.ma.masked_invalid(out[0]).compressed()
            if src.nodata is not None:
                vals = vals[vals != src.nodata]
            if vals.size:
                return vals, radius
        except ValueError:  # buffer outside the raster
            pass
        radius *= 2
    return np.array([]), np.nan


def site_buffer_max(raster: Path | np.ndarray, sites: gpd.GeoDataFrame, radius_m: float,
                    radius_max_m: float, profile: dict | None = None) -> pd.DataFrame:
    """Maximum of a raster in a buffer around each site (Section 2.6).

    ``raster`` is a file or an array (then ``profile`` is required).
    Returns a table with the value and the radius actually used.
    """
    def run(src):
        rows = []
        for geom in sites.to_crs(src.crs).geometry:
            vals, radius = _buffer_values(src, geom, radius_m, radius_max_m)
            rows.append((float(vals.max()) if vals.size else np.nan, radius))
        return pd.DataFrame(rows, columns=["value", "radius_m"], index=sites.index)

    if isinstance(raster, np.ndarray):
        prof = profile.copy()
        prof.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0)
        with MemoryFile() as mem, mem.open(**prof) as dst:
            dst.write(np.where(np.isfinite(raster), raster, -9999.0).astype(np.float32), 1)
            return run(dst)
    with rasterio.open(raster) as src:
        return run(src)


def site_pixel_index(valid: np.ndarray, profile: dict, sites: gpd.GeoDataFrame,
                     radius_m: float, radius_max_m: float) -> list[np.ndarray]:
    """For each site, the positions (in ``array[valid]``) of the pixels of its buffer.

    Used when the Hazard Index is recomputed many times (sensitivity analysis):
    the buffer is found once, then each scenario costs one maximum per site.
    Same rule as :func:`site_buffer_max`.
    """
    position = np.full(valid.size, -1, dtype=np.int64)
    position[np.flatnonzero(valid.ravel())] = np.arange(int(valid.sum()))
    index_raster = np.where(valid.ravel(), np.arange(valid.size), -1).reshape(valid.shape)

    prof = profile.copy()
    prof.update(driver="GTiff", dtype="int32", count=1, nodata=-1)
    out = []
    with MemoryFile() as mem, mem.open(**prof) as dst:
        dst.write(index_raster.astype(np.int32), 1)
        for geom in sites.to_crs(dst.crs).geometry:
            vals, _ = _buffer_values(dst, geom, radius_m, radius_max_m)
            out.append(position[vals.astype(np.int64)] if vals.size else np.array([], dtype=np.int64))
    return out
