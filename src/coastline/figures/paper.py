"""Publication figures of the article (Open Research Europe).

One function per figure, numbered as in the article (Draft v3):

    fig2  workflow diagram
    fig3  cascade weights: correlation network and weights
    fig4  Hazard Index, continuous and classified        (preview; the published map is made in ArcGIS Pro)
    fig5  Vulnerability, Exposure and Risk classes        (preview; ArcGIS Pro)
    fig6  ranking of the 34 sites: H, V, E and Risk classes
    fig7  OAT stability map, pixel level                  (preview; ArcGIS Pro)
    fig8  OAT, Hazard rank range of the sites
    fig9  Risk class of each site under the sensitivity tests
    ext   the eight normalized Hazard factors (Extended data Figure 1)

Inputs are the outputs of scripts 01-05 (results folder). Outputs:
    results/figures/<name>.png / .svg / .pdf   publication style (16 cm wide, 300 dpi)
    results/figures/arcgis/<folder>/           layers for ArcGIS Pro + README with the symbology

Class colours are the shared palette (green - yellow - red); every figure adds a second cue
(class letter in the tile, symbol size) for readers with red-green colour blindness.
"""
import os
import math
import shutil
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import geopandas as gpd

from ..config import load_config

# ============================================================================ CONFIGURATION
CFG = load_config()
FIG_DIR = CFG.results_dir / "figures"
MAPS_DIR = FIG_DIR / "arcgis"                      # layers and symbology notes for ArcGIS Pro
UTM = CFG.crs
ISLAND_BOX = (410000, 3800000, 670000, 3980000)      # Cyprus in UTM 36N (xmin, ymin, xmax, ymax)
INCLUDE_MEAN_RANK = False    # mean-rank aggregation in fig9: off (see docs/VERIFICATION.md)


def result(*parts):
    """Path of an output of scripts 01-05."""
    return CFG.results_dir.joinpath(*parts)


def load_weights():
    """Cascade weights of Table 5 (3 decimals), from script 01."""
    return pd.read_csv(result("hazard", "weights.csv")).set_index("factor")["weight_rounded"].to_dict()


# ============================================================================ PALETTE AND STYLE
CLASS3 = {1: "#1B9E77", 2: "#F2C12E", 3: "#D62F25"}
LABEL3 = {1: "Low", 2: "Medium", 3: "High"}
CLASS5 = {1: "#1B9E77", 2: "#A6D96A", 3: "#F2C12E", 4: "#F58A2E", 5: "#D62F25"}
LABEL5 = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}
EDGE = "#333333"
GREY = "#8C8C8C"
LAND = "#F1F1EF"
CM = 1 / 2.54
W_FULL = 16 * CM
SIZE3 = {1: 14, 2: 28, 3: 52}
SIZE5 = {1: 8, 2: 16, 3: 27, 4: 40, 5: 56}
SHORT = {"3": "L", "1": "L"}
 
 
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 120,
    })
 
 
def text_on(hexcolor):
    r, g, b = [int(hexcolor[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else "black"
 
 
def panel_label(ax, label, x=0.0, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
 
 
def save(fig, name, copy_to=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300)
    if copy_to is not None:
        Path(copy_to).mkdir(parents=True, exist_ok=True)
        shutil.copy(FIG_DIR / f"{name}.png", Path(copy_to) / f"{name}_preview.png")
    plt.close(fig)
    print(f"  figure saved: {FIG_DIR / (name + '.png')}")
 
 
# ============================================================================ DATA
def clean_name(n):
    return n.replace("Archeological", "Archaeological")


def sites_df():
    """One row per site: classes and values of the indices, UTM coordinates."""
    r = pd.read_csv(result("risk", "risk_index.csv"))
    p = gpd.read_file(CFG.input("site_points")).to_crs(UTM)
    xy = pd.DataFrame({"id": p["id"], "x_utm36n": p.geometry.x, "y_utm36n": p.geometry.y,
                       "district": p["district"] if "district" in p else ""})
    s = r.merge(xy, on="id").rename(columns={"hi": "HI", "vi_norm": "VI_norm", "vi_resc": "VI_resc",
                                             "e_norm": "E_norm", "r": "R"})
    s["Hc"], s["Vc"], s["Ec"], s["Rc"] = s.hi_class, s.vi_class, s.e_class, s.r_class
    s["name"] = s["name"].map(clean_name)
    return s.sort_values("id").reset_index(drop=True)


def mot(values, n):
    from skimage.filters import threshold_multiotsu
    v = np.asarray(values, float)
    t = threshold_multiotsu(v[np.isfinite(v)], classes=n)
    return np.digitize(v, t) + 1, t
 
 
def short_name(n):
    rep = {"Lempa-Lakkoi Archaeological Site": "Lempa-Lakkoi", "Old Catacombs (south) Phafos": "Old Catacombs (south)",
           "Choirokoitia (Khirokitia)": "Choirokoitia", "Dromolaxia - Byzakia Excavation Site": "Dromolaxia-Byzakia",
           "Archaeological Site of ": "", " Archaeological Site": "",
           "Archaeological site of ": "", "Archaeological Site": "Site"}
    for k, v in rep.items():
        n = n.replace(k, v)
    return n.strip()
 
 
# ============================================================================ MAP HELPERS
def get_outline():
    """Island outline in UTM 36N. Built once from the EuroRegionalMap coastline and cached."""
    if os.environ.get("COASTLINE_NO_OUTLINE"):
        return None
    cache = MAPS_DIR / "_common" / "Cyprus_outline.gpkg"
    if cache.exists():
        return gpd.read_file(cache)
    gdf = None
    try:
        from shapely.geometry import box
        from shapely.ops import polygonize, unary_union
        gpkg = CFG.input("erm_gpkg")
        if gpkg.exists():
            bx = gpd.GeoSeries([box(*ISLAND_BOX)], crs=UTM)
            lines = gpd.read_file(gpkg, layer=CFG.raw["inputs"]["coastline_layer"], bbox=bx).to_crs(UTM)
            polys = [p for p in polygonize(unary_union(list(lines.geometry))) if 1e6 < p.area < 2e10]
            gdf = gpd.GeoDataFrame(geometry=polys, crs=UTM)
    except Exception as exc:
        print(f"  coastline layer could not be read ({exc})")
    if gdf is None or len(gdf) == 0:
        import glob
        hits = glob.glob("/usr/**/naturalearth_lowres.shp", recursive=True)
        if hits:
            w = gpd.read_file(hits[0])
            gdf = w[w["name"].isin(["Cyprus", "N. Cyprus"])].to_crs(UTM)[["geometry"]]
            print("  PREVIEW ONLY: coarse Natural Earth outline used (coastline file not found), not cached")
            return gdf
    if gdf is None or len(gdf) == 0:
        print("  no outline available: maps are drawn without island outline")
        return None
    (MAPS_DIR / "_common").mkdir(parents=True, exist_ok=True)
    gdf.to_file(cache, driver="GPKG")
    gdf.to_file(MAPS_DIR / "_common" / "Cyprus_outline.shp")
    return gdf
 
 
def island_extent(outline, s, pad=6000):
    if outline is not None and len(outline):
        x0, y0, x1, y1 = outline.total_bounds
    else:
        x0, y0, x1, y1 = s.x_utm36n.min() - 20000, s.y_utm36n.min() - 20000, s.x_utm36n.max() + 20000, s.y_utm36n.max() + 20000
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad
 
 
SEA = "#FBFAF9"
GRID = "#CFCFCF"
 
 
def base_map(ax, outline, extent):
    """ArcGIS-like frame: off-white background, grey graticule, thin dark coastline, outward ticks."""
    ax.set_facecolor(SEA)
    if outline is not None:
        outline.plot(ax=ax, facecolor=LAND, edgecolor="none", zorder=0.2)
        outline.boundary.plot(ax=ax, color="#555555", linewidth=0.5, zorder=4.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#555555")
    graticule(ax)
 
 
def _dm(v, pos, neg):
    h = pos if v >= 0 else neg
    v = abs(v)
    d = int(v + 1e-9)
    m = int(round((v - d) * 60))
    return f"{d}\u00b0{h}" if m == 0 else f"{d}\u00b0{m:02d}\u2032{h}"
 
 
def graticule(ax):
    from pyproj import Transformer
    fwd = Transformer.from_crs(4326, 32636, always_xy=True)
    inv = Transformer.from_crs(32636, 4326, always_xy=True)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lon0, lat0 = inv.transform(x0, y0)
    lon1, lat1 = inv.transform(x1, y1)
    step = 0.5 if (lon1 - lon0) > 1.6 else 0.25
    lons = np.arange(math.ceil(lon0 / step) * step, lon1, step)
    lats = np.arange(math.ceil(lat0 / step) * step, lat1, step)
    lm = (lat0 + lat1) / 2
    ln = (lon0 + lon1) / 2
    ax.set_xticks([fwd.transform(l, lm)[0] for l in lons])
    ax.set_xticklabels([_dm(l, "E", "W") for l in lons])
    ax.set_yticks([fwd.transform(ln, l)[1] for l in lats])
    ax.set_yticklabels([_dm(l, "N", "S") for l in lats], rotation=90, va="center")
    mlons = np.arange(math.ceil(lon0 / (step / 2)) * (step / 2), lon1, step / 2)
    mlats = np.arange(math.ceil(lat0 / (step / 2)) * (step / 2), lat1, step / 2)
    ax.set_xticks([fwd.transform(l, lm)[0] for l in mlons], minor=True)
    ax.set_yticks([fwd.transform(ln, l)[1] for l in mlats], minor=True)
    ax.tick_params(which="major", length=3.5, width=0.7, pad=2, labelsize=6.5, color="#555555")
    ax.tick_params(which="minor", length=2, width=0.5, color="#555555")
    ax.grid(True, which="major", color=GRID, lw=0.5, ls="-", zorder=0.5)
    ax.set_axisbelow(True)
 
 
def scalebar(ax, km=50, frac=(0.60, 0.07), north=False):
    """Segmented black-and-white scale bar with 'Kilometers' below, as in ArcGIS Pro. Optional north arrow above."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx, sy = x1 - x0, y1 - y0
    x = x0 + frac[0] * sx
    y = y0 + frac[1] * sy
    seg = km * 1000 / 4
    h = 0.016 * sy
    for i in range(4):
        ax.add_patch(Rectangle((x + i * seg, y), seg, h, fc="black" if i % 2 == 0 else "white", ec="black", lw=0.6, zorder=8))
    ax.text(x, y + h + 0.012 * sy, "0", fontsize=6.5, ha="center", va="bottom", zorder=8)
    ax.text(x + 4 * seg, y + h + 0.012 * sy, f"{km}", fontsize=6.5, ha="center", va="bottom", zorder=8)
    ax.text(x + 2 * seg, y - 0.012 * sy, "Kilometers", fontsize=6.5, ha="center", va="top", zorder=8)
    if north:
        cx = x + 2 * seg
        yb = y + h + 0.075 * sy
        w, hh = 0.034 * sx, 0.085 * sy
        ax.add_patch(plt.Polygon([(cx, yb + hh), (cx - w / 2, yb), (cx, yb + 0.32 * hh), (cx + w / 2, yb)],
                                 closed=True, fc="black", ec="black", lw=0.5, zorder=9))
        ax.text(cx, yb + hh + 0.01 * sy, "N", fontsize=7.5, ha="center", va="bottom", family="serif", zorder=9)
 
 
def north_arrow(ax, pos=(0.95, 0.80)):
    """Kept for compatibility: use scalebar(..., north=True) to place the arrow above the scale bar."""
    return None
 
 
def draw_sites(ax, df, col, palette, sizes, edge=EDGE, lw=0.5, zbase=3):
    for k in sorted(palette):
        d = df[df[col] == k]
        if len(d):
            ax.scatter(d.x_utm36n, d.y_utm36n, s=sizes[k], c=palette[k], edgecolors=edge,
                       linewidths=lw, zorder=zbase + k)
 
 
def legend_handles(palette, labels, sizes, counts=None, title=None):
    hs = []
    for k in sorted(palette):
        lab = labels[k] + (f" ({counts.get(k, 0)})" if counts is not None else "")
        hs.append(Line2D([0], [0], marker="o", lw=0, mfc=palette[k], mec=EDGE, mew=0.5,
                         ms=math.sqrt(sizes[k]), label=lab))
    return hs
 
 
def write_sites_shp(folder):
    s = sites_df()
    g = gpd.GeoDataFrame({
        "id": s.id, "name": s.name, "district": s.district,
        "HI_val": s.HI.round(4), "HI_cls": s.Hc, "HI_lbl": s.Hc.map(LABEL5),
        "VI_cls": s.Vc, "VI_lbl": s.Vc.map(LABEL3), "VI_resc": s.VI_resc,
        "E_cls": s.Ec, "E_lbl": s.Ec.map(LABEL5), "reviews": s.reviews,
        "R_val": s.R, "R_cls": s.Rc, "R_lbl": s.Rc.map(LABEL3),
        "label": np.where(s.Rc == 3, s.name.map(short_name) + " (" + s.R.astype(str) + ")", ""),
    }, geometry=gpd.points_from_xy(s.x_utm36n, s.y_utm36n), crs=UTM)
    Path(folder).mkdir(parents=True, exist_ok=True)
    g.to_file(Path(folder) / "Sites_HVER.shp")
 
 
def copy_outline(folder):
    out = get_outline()
    if out is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        out.to_file(Path(folder) / "Cyprus_outline.shp")
 
 
def write_readme(folder, text):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "README_ArcGIS.txt").write_text(text, encoding="utf-8")
 
 
PALETTE_TXT = ("Palette (hex)\n  3 classes (Vulnerability, Risk): Low #1B9E77, Medium #F2C12E, High #D62F25\n"
               "  5 classes (Hazard, Exposure): Very Low #1B9E77, Low #A6D96A, Medium #F2C12E, High #F58A2E, Very High #D62F25\n"
               "  Outline of symbols and island: #333333 / #8C8C8C, land fill #F1F1EF\n"
               "Layout: 16 cm wide, text Arial 8 pt, scale bar, north arrow, light graticule in degrees, legend outside the frame.\n"
               "Export: PNG 300 dpi and PDF. The Python preview in Images\\Figures shows the target design.\n")
 
 
# ============================================================================ FIG 2: workflow
def fig2():
    set_style()
    fig = plt.figure(figsize=(W_FULL, 12.4 * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 76)
    ax.axis("off")
    dark = {"H": "#4D4D4D", "V": "#4D4D4D", "E": "#4D4D4D", "R": "#4D4D4D", "S": "#4D4D4D"}
    tint = {"H": "#F2F2F2", "V": "#F2F2F2", "E": "#F2F2F2", "R": "#E4E4E4", "S": "#FFFFFF"}
 
    def rbox(x, y, w, h, fc, ec=EDGE, lw=0.6, ls="-", z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
 
    def stage(x, y, w, h, key, title, detail):
        rbox(x, y, w, h, tint[key], ec=dark[key], lw=0.7)
        ax.text(x + w / 2, y + h - 2.7, title, ha="center", va="center", fontsize=6.9, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + (h - 3.0) / 2, detail, ha="center", va="center", fontsize=6.2, zorder=3, linespacing=1.3)
 
    def arrow(x0, y0, x1, y1, ls="-", color=EDGE):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, mutation_scale=7, linestyle=ls,
                                    shrinkA=0, shrinkB=0), zorder=1)
 
    cw, bh = 21.0, 11.0
    xs = {"H": 6.5, "V": 29.5, "E": 52.5}
    heads = {"H": "HAZARD", "V": "VULNERABILITY", "E": "EXPOSURE"}
    tops = [66.0 - i * 13.5 for i in range(4)]
    rows = ["INPUT", "PROCESSING", "INDEX", "CLASSES"]
    content = {
        "H": [("Eight open datasets", "SAI, BUI, LSI, CPI,\nFFI, RNI, SLRI, FHI"),
              ("Weighted combination", "Normalisation to [0, 1]\nCascade weights"),
              ("Hazard Index", "30 m surface; maximum\nwithin 125 m of each site"),
              ("Hazard class", "Five classes\n(Multi-Otsu)")],
        "V": [("Seven attributes", "Heritage status, Enclosure,\nShelter, Land Use, Elevation,\nDistance to Coast, Slope"),
              ("Scoring", "Score 1 to 3 per attribute\n(Multi-Otsu for continuous)"),
              ("Vulnerability Index", "Mean score,\nrescaled to [0, 1]"),
              ("Vulnerability class", "Three classes (Multi-Otsu),\nrescaled to 1, 3 and 5")],
        "E": [("Google Places", "Review count\nof each site"),
              ("Transformation", "Logarithm of the\nreview count"),
              ("Exposure Index", "Rescaled to [0, 1]"),
              ("Exposure class", "Five classes\n(Multi-Otsu)")],
    }
    for key in "HVE":
        x = xs[key]
        rbox(x, 69.0, cw, 5.0, dark[key], ec=dark[key])
        ax.text(x + cw / 2, 71.5, heads[key], ha="center", va="center", fontsize=7.2, fontweight="bold", color="white", zorder=3)
        for i, (ti, de) in enumerate(content[key]):
            stage(x, tops[i] - bh, cw, bh, key, ti, de)
            if i < 3:
                arrow(x + cw / 2, tops[i] - bh, x + cw / 2, tops[i + 1] + 0.1)
    for i, r in enumerate(rows):
        ax.text(3.3, tops[i] - bh / 2, r, rotation=90, ha="center", va="center", fontsize=5.8, color="#666666", fontweight="bold")
    # separators between the three indices and the sensitivity panel
    ax.plot([76.0, 76.0], [9.5, 74.0], color="#BBBBBB", lw=0.6, ls=":", zorder=0)
    # risk
    rbox(12.0, 1.0, 56.0, 8.2, tint["R"], ec=dark["R"], lw=0.9)
    ax.text(40.0, 6.6, "Risk = Hazard class  ×  Vulnerability class  ×  Exposure class", ha="center", va="center",
            fontsize=7.0, fontweight="bold", zorder=3)
    ax.text(40.0, 3.4, "Three classes (Multi-Otsu): Low, Medium, High", ha="center", va="center", fontsize=6.2, zorder=3)
    for key in "HVE":
        arrow(xs[key] + cw / 2, tops[3] - bh, xs[key] + cw / 2, 9.3)
    # sensitivity panel
    sx, sw = 78.0, 21.5
    rbox(sx, 69.0, sw, 5.0, dark["S"], ec=dark["S"])
    ax.text(sx + sw / 2, 71.5, "SENSITIVITY ANALYSES", ha="center", va="center", fontsize=6.8, fontweight="bold", color="white", zorder=3)
    tests = [("Weights", ["One-at-a-time, ±20%", "32 weight vectors"]),
             ("Vulnerability", ["Site polygon", "No Enclosure and Shelter", "Fixed thresholds"]),
             ("Aggregation", ["Product of indices", "Mean rank"]),
             ("Coastal inputs", ["Without CPI", "Also without Distance\nto Coast", "Also without Elevation"])]
    hts = [11.0, 14.0, 11.0, 17.0]
    y = 66.0
    for (ti, items), h in zip(tests, hts):
        rbox(sx, y - h, sw, h, tint["S"], ec=dark["S"], ls="--", lw=0.6)
        ax.text(sx + 1.6, y - 2.6, ti, ha="left", va="center", fontsize=6.8, fontweight="bold", zorder=3)
        yy = y - 5.6
        for it in items:
            nl = it.count("\n") + 1
            ax.text(sx + 1.6, yy + 1.1, "•  " + it.replace("\n", "\n    "), ha="left", va="top",
                    fontsize=6.0, zorder=3, linespacing=1.25)
            yy -= 3.1 + (nl - 1) * 2.3
        y -= h + 2.2
    ax.plot([sx + sw / 2, sx + sw / 2], [y + 2.2, 4.8], color="#555555", lw=0.8, ls=(0, (3, 2)), zorder=1)
    arrow(sx + sw / 2, 4.8, 68.2, 4.8, ls=(0, (3, 2)), color="#555555")
    save(fig, "Fig02_workflow")
 
 
# ============================================================================ FIG 4: Hazard Index maps
def read_raster(path, step_target=3200):
    import rasterio
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            a[a == src.nodata] = np.nan
        b = src.bounds
        prof = src.profile
    k = max(1, int(math.ceil(a.shape[1] / step_target)))
    return a, (b.left, b.right, b.bottom, b.top), prof, k
 
 
def fig4():
    set_style()
    import rasterio
    hi_tif = result("hazard", "hazard_index.tif")
    a, ext, prof, k = read_raster(hi_tif)
    valid = np.isfinite(a)
    cls, thr = mot(a[valid], 5)
    c = np.zeros(a.shape, "uint8")
    c[valid] = cls
    counts = {i: float((cls == i).mean() * 100) for i in range(1, 6)}
    folder = MAPS_DIR / "Fig04_Hazard"
    folder.mkdir(parents=True, exist_ok=True)
    p = prof.copy()
    p.update(dtype="uint8", nodata=0, count=1, compress="lzw")
    with rasterio.open(folder / "HI_classified_5.tif", "w", **p) as dst:
        dst.write(c, 1)
    (folder / "HI_thresholds.txt").write_text(
        "Multi-Otsu thresholds of the Hazard Index surface (5 classes, valid pixels)\n" +
        ", ".join(f"{t:.4f}" for t in thr) + "\n" +
        "Share of pixels per class (%): " + ", ".join(f"{LABEL5[i]} {counts[i]:.1f}" for i in range(1, 6)) + "\n", encoding="utf-8")
    write_sites_shp(folder)
    copy_outline(folder)
    shutil.copy(hi_tif, folder / hi_tif.name)
 
    s = sites_df()
    outline = get_outline()
    cmap = LinearSegmentedColormap.from_list("hazard", [CLASS5[i] for i in range(1, 6)])
    cmap5 = ListedColormap([CLASS5[i] for i in range(1, 6)])
    cmap5.set_bad((0, 0, 0, 0))
    cmap.set_bad((0, 0, 0, 0))
    ratio = (ext[3] - ext[2]) / (ext[1] - ext[0])
    mapw = 12.2 * CM
    maph = mapw * ratio
    fig = plt.figure(figsize=(W_FULL, 2 * maph + 3.2 * CM))
    left, wmap = 0.075, mapw / W_FULL
    hfrac = maph / fig.get_figheight()
    ax1 = fig.add_axes([left, 0.55 + (0.45 - hfrac) / 2 - 0.02, wmap, hfrac])
    ax2 = fig.add_axes([left, 0.06 + (0.45 - hfrac) / 2 - 0.02, wmap, hfrac])
    extent = island_extent(outline, s) if outline is not None else (ext[0], ext[1], ext[2], ext[3])
    vmin, vmax = np.nanmin(a), np.nanmax(a)
    im1 = ax1.imshow(np.ma.masked_invalid(a[::k, ::k]), extent=ext, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest", zorder=1)
    cm = np.ma.masked_where(c == 0, c)
    ax2.imshow(cm[::k, ::k], extent=ext, origin="upper", cmap=cmap5, vmin=0.5, vmax=5.5, interpolation="nearest", zorder=1)
    for ax in (ax1, ax2):
        base_map(ax, outline, extent)
        for im in ax.images:
            im.set_zorder(1)
    ax1.scatter(s.x_utm36n, s.y_utm36n, s=14, c="white", edgecolors=EDGE, linewidths=0.6, zorder=6)
    ax2.scatter(s.x_utm36n, s.y_utm36n, s=18, c=[CLASS5[v] for v in s.Hc], edgecolors="black", linewidths=0.7, zorder=6)
    for ax in (ax1, ax2):
        scalebar(ax, 50, north=True)
    panel_label(ax1, "(a)")
    panel_label(ax2, "(b)")
    # colourbar and legend, right
    cax = fig.add_axes([left + wmap + 0.04, 0.62, 0.022, 0.28])
    cb = fig.colorbar(im1, cax=cax)
    cb.set_label("Hazard Index", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    hs = [Patch(fc=CLASS5[i], ec=EDGE, lw=0.4, label=f"{LABEL5[i]}") for i in range(5, 0, -1)]
    hs.append(Line2D([0], [0], marker="o", lw=0, mfc="white", mec="black", ms=4.5, label="Site"))
    fig.legend(handles=hs, loc="center left", bbox_to_anchor=(left + wmap + 0.03, 0.27), frameon=False,
               title="Hazard class", fontsize=7, title_fontsize=7, alignment="left")
    save(fig, "Fig04_Hazard_Index", copy_to=folder)
    (folder / "README_ArcGIS.txt").write_text(
        "FIG 4 - Hazard Index, continuous (a) and classified (b)\n\n"
        "Layers: hazard_index.tif (continuous), HI_classified_5.tif (classified, values 1-5, 0 = no data),\n"
        "Sites_HVER.shp (fields HI_cls, HI_lbl), Cyprus_outline.shp.\n"
        "Panel (a): stretch min-max, colour ramp with the five class colours as stops (see palette).\n"
        "Panel (b): unique values 1-5 with the five class colours. Sites: fill = HI_cls, black outline, constant size.\n"
        "Thresholds and class shares: HI_thresholds.txt. The two panels share extent and outline.\n\n" + PALETTE_TXT,
        encoding="utf-8")
 
 
# ============================================================================ FIG 3: cascade weights network
def fig3():
    set_style()
    WEIGHTS = load_weights()
    raw = pd.read_csv(result("hazard", "correlation.csv"), index_col=0)
    codes = list(WEIGHTS)
    C = raw.loc[codes, codes].astype(float)
    ang = np.pi / 2 - 2 * np.pi * np.arange(8) / 8
    pos = {c: (np.cos(a), np.sin(a)) for c, a in zip(codes, ang)}
    fig = plt.figure(figsize=(W_FULL, 7.6 * CM))
    ax = fig.add_axes([0.0, 0.02, 0.56, 0.92])
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    pairs = [(a, b) for i, a in enumerate(codes) for b in codes[i + 1:]]
    top = sorted(pairs, key=lambda p: -abs(C.loc[p]))[:3]
    for a, b in pairs:
        r = C.loc[a, b]
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#555555" if r > 0 else "#B0463C",
                lw=0.3 + 12 * abs(r), ls="-" if r > 0 else (0, (3, 1.5)), alpha=0.75, solid_capstyle="round", zorder=1)
    for a, b in top:
        r = C.loc[a, b]
        mx, my = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        ax.text(mx, my, f"{r:+.2f}".replace("+", ""), fontsize=6.2, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    for c in codes:
        x, y = pos[c]
        ax.scatter([x], [y], s=3600 * WEIGHTS[c], c="#DCE3EC", edgecolors=EDGE, linewidths=0.7, zorder=3)
        ax.text(x * 1.50, y * 1.40, f"{c}\n{WEIGHTS[c]:.3f}", ha="center", va="center", fontsize=6.8, zorder=5, linespacing=1.15)
    panel_label(ax, "(a)", x=0.02, y=0.96)
    hs = [Line2D([0], [0], color="#555555", lw=3, label="Positive correlation"),
          Line2D([0], [0], color="#B0463C", lw=3, ls=(0, (3, 1.5)), label="Negative correlation")]
    ax.legend(handles=hs, loc="lower left", frameon=False, fontsize=6.2, bbox_to_anchor=(-0.02, -0.04))
    ax2 = fig.add_axes([0.66, 0.2, 0.31, 0.68])
    order = sorted(WEIGHTS, key=lambda c: -WEIGHTS[c])
    ax2.barh(range(8), [WEIGHTS[c] for c in order], color="#DCE3EC", edgecolor=EDGE, lw=0.6, height=0.7)
    for i, c in enumerate(order):
        ax2.text(WEIGHTS[c] + 0.005, i, f"{WEIGHTS[c]:.3f}", va="center", fontsize=6.8)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels(order)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 0.32)
    ax2.set_xlabel("Cascade weight")
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    panel_label(ax2, "(b)", x=-0.30, y=1.04)
    save(fig, "Fig03_Cascade_weights")
 
 
# ============================================================================ FIG 6: ranking matrix
def fig6():
    set_style()
    s = sites_df().sort_values(["R", "HI"], ascending=[False, False]).reset_index(drop=True)
    n = len(s)
    rowh = 0.335
    fig_h = n * rowh + 2.7
    fig = plt.figure(figsize=(W_FULL, fig_h * CM))
    ax = fig.add_axes([0.25, 0.3 / fig_h, 0.74, (n * rowh + 1.9) / fig_h])
    ax.set_xlim(-0.1, 8.6)
    ax.set_ylim(-0.7, n + 3.4)
    ax.axis("off")
    xcols = [0.0, 1.0, 2.0, 3.0]
    heads = ["Hazard", "Vulnerability", "Exposure", "Risk"]
    for x, h in zip(xcols, heads):
        ax.text(x + 0.45, n - 0.1, h, rotation=50, ha="left", va="bottom", fontsize=7)
    ax.text(4.4, n - 0.1, "Risk value (R)", ha="left", va="bottom", fontsize=7)
    for i, r in s.iterrows():
        y = n - 1 - i
        tiles = [(r.Hc, CLASS5, LABEL5[r.Hc]), (r.Vc, CLASS3, LABEL3[r.Vc]), (r.Ec, CLASS5, LABEL5[r.Ec]), (r.Rc, CLASS3, LABEL3[r.Rc])]
        for x, (v, pal, t) in zip(xcols, tiles):
            ax.add_patch(Rectangle((x, y - 0.42), 0.9, 0.84, fc=pal[v], ec="white", lw=0.5))
            ax.text(x + 0.45, y, t, ha="center", va="center", fontsize=5.8, color=text_on(pal[v]))
        ax.text(-0.25, y, short_name(r['name']), ha="right", va="center", fontsize=6.6)
        w = 2.9 * r.R / 100
        ax.add_patch(Rectangle((4.4, y - 0.30), max(w, 0.04), 0.60, fc=CLASS3[r.Rc], ec=EDGE, lw=0.4))
        ax.text(4.4 + w + 0.1, y, f"{int(r.R)}", ha="left", va="center", fontsize=6.4)
    cut = [i for i in range(n - 1) if s.Rc[i] != s.Rc[i + 1]]
    for i in cut:
        y = n - 1 - i - 0.5
        ax.plot([-0.1, 8.6], [y, y], color=GREY, lw=0.6, ls="--")
    for k in (3, 2, 1):
        idx = s.index[s.Rc == k]
        ymid = n - 1 - (idx.min() + idx.max()) / 2
        ax.text(8.55, ymid, f"{LABEL3[k]} Risk\n({len(idx)} sites)", ha="right", va="center", fontsize=6.6,
                color="black", rotation=0, bbox=dict(fc="white", ec=CLASS3[k], lw=1.0, boxstyle="round,pad=0.25"))
    save(fig, "Fig06_Ranking_matrix")
 
 
# ============================================================================ FIG 5: V, E, R maps
def fig5():
    set_style()
    s = sites_df()
    outline = get_outline()
    folder = MAPS_DIR / "Fig05_VER"
    write_sites_shp(folder)
    copy_outline(folder)
    extent = island_extent(outline, s)
    ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])
    pw = 6.8 * CM
    ph = pw * ratio
    figh = 2 * ph + 4.6 * CM
    fig = plt.figure(figsize=(W_FULL, figh))
    gx = [0.075, 0.565]
    wfrac = pw / W_FULL
    hfrac = ph / figh
    ytop = 1 - 0.035 - hfrac
    ybot = ytop - hfrac - 0.15
    axes = {"V": fig.add_axes([gx[0], ytop, wfrac, hfrac]), "E": fig.add_axes([gx[1], ytop, wfrac, hfrac]),
            "R": fig.add_axes([gx[0], ybot, wfrac, hfrac]), "Z": fig.add_axes([gx[1], ybot, wfrac, hfrac])}
    for k in "VER":
        base_map(axes[k], outline, extent)
        scalebar(axes[k], 50, north=(k == "V"))
    draw_sites(axes["V"], s, "Vc", CLASS3, SIZE3)
    draw_sites(axes["E"], s, "Ec", CLASS5, SIZE5)
    draw_sites(axes["R"], s, "Rc", CLASS3, SIZE3)
    cV = s.Vc.value_counts().to_dict()
    cE = s.Ec.value_counts().to_dict()
    cR = s.Rc.value_counts().to_dict()
    axes["V"].legend(handles=legend_handles(CLASS3, LABEL3, SIZE3, cV), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
                     frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=0.8)
    axes["E"].legend(handles=legend_handles(CLASS5, LABEL5, SIZE5, cE), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
                     frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=0.8)
    axes["R"].legend(handles=legend_handles(CLASS3, LABEL3, SIZE3, cR), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
                     frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=0.8)
    for k, lab in zip("VER", ("(a) Vulnerability", "(b) Exposure", "(c) Risk")):
        pass
    panel_label(axes["V"], "(a) Vulnerability")
    panel_label(axes["E"], "(b) Exposure")
    panel_label(axes["R"], "(c) Risk")
    # detail Paphos - Limassol
    hi = s[s.Rc == 3]
    x0, x1 = hi.x_utm36n.min() - 14000, hi.x_utm36n.max() + 22000
    cy = (hi.y_utm36n.min() + hi.y_utm36n.max()) / 2
    half = (x1 - x0) * ratio / 2
    zext = (x0, x1, cy - half, cy + half)
    z = axes["Z"]
    base_map(z, outline, zext)
    zs = s[(s.x_utm36n.between(x0, x1)) & (s.y_utm36n.between(zext[2], zext[3]))]
    draw_sites(z, zs, "Rc", CLASS3, {k: v * 1.5 for k, v in SIZE3.items()})
    hs = hi.sort_values("y_utm36n", ascending=False).reset_index(drop=True)
    for i, r in hs.iterrows():
        z.annotate(f"{short_name(r['name'])} ({int(r.R)})", xy=(r.x_utm36n, r.y_utm36n),
                   xytext=(0.50, 0.92 - 0.085 * i), textcoords="axes fraction", fontsize=6.2, ha="left", va="center",
                   zorder=12, annotation_clip=False,
                   arrowprops=dict(arrowstyle="-", lw=0.5, color="#222222", shrinkA=0, shrinkB=3),
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#BBBBBB", lw=0.4, alpha=0.95))
    scalebar(z, 20, frac=(0.60 + 1.0 / 6.8, 0.07))   # shifted 1.0 cm right (panel width 6.8 cm)
    panel_label(z, "(d) Risk, Paphos and Limassol")
    z.legend(handles=legend_handles(CLASS3, LABEL3, SIZE3), loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
             frameon=False, fontsize=6.4, handletextpad=0.2, columnspacing=0.8, title="Risk class", title_fontsize=6.4)
    save(fig, "Fig05_V_E_R_maps", copy_to=folder)
    write_readme(folder,
                 "FIG 5 - Vulnerability (a), Exposure (b), Risk (c) and detail Paphos-Limassol (d)\n\n"
                 "Layers: Sites_HVER.shp (VI_cls/VI_lbl, E_cls/E_lbl, R_cls/R_lbl, R_val, label), Cyprus_outline.shp.\n"
                 "(a) Unique values on VI_cls, colours of the 3-class palette, symbol size 14/28/52 (points^2).\n"
                 "(b) Unique values on E_cls, colours of the 5-class palette, symbol size 8/16/27/40/56.\n"
                 "(c) Unique values on R_cls, 3-class palette, size 14/28/52, High drawn on top (symbol level drawing).\n"
                 "(d) Same symbology as (c), size x1.5, extent around the five High sites, label field 'label' (name and R) for High only.\n"
                 "Legends outside the frame with the class counts: V 10/7/17, E 3/10/10/8/3, R 17/12/5.\n\n" + PALETTE_TXT)
 
 
# ============================================================================ FIG 7: OAT stability map
def fig7():
    set_style()
    import rasterio
    with rasterio.open(result("sensitivity", "oat_stability.tif")) as src:
        pct = src.read(1).astype("float32")
        pct[pct == src.nodata] = np.nan
        prof = src.profile
        b = src.bounds
    valid = np.isfinite(pct)
    n_scen = 320                                     # 8 factors x 40 perturbed weights
    count = np.rint(pct[valid] * n_scen / 100).astype("uint16")
    thr = pd.read_csv(result("hazard", "hazard_class_thresholds.csv"))["threshold"].values
    folder = MAPS_DIR / "Fig07_OAT_stability"
    folder.mkdir(parents=True, exist_ok=True)
    brk = [0, 5, 20, 50]
    scl = np.zeros(valid.shape, "uint8")
    scl[valid] = 1 + np.digitize(pct[valid], [1e-9, 5, 20, 50], right=True)
    cnt_r = np.zeros(valid.shape, "uint16")
    cnt_r[valid] = count
    p = prof.copy()
    p.update(count=1, compress="lzw")
    with rasterio.open(folder / "OAT_changes_count.tif", "w", **{**p, "dtype": "uint16", "nodata": 65535}) as dst:
        out = np.full(valid.shape, 65535, "uint16")
        out[valid] = count
        dst.write(out, 1)
    with rasterio.open(folder / "OAT_changes_percent.tif", "w", **{**p, "dtype": "float32", "nodata": -9999.0}) as dst:
        dst.write(np.where(valid, pct, -9999.0).astype("float32"), 1)
    with rasterio.open(folder / "OAT_stability_class.tif", "w", **{**p, "dtype": "uint8", "nodata": 0}) as dst:
        dst.write(scl, 1)
    write_sites_shp(folder)
    copy_outline(folder)
    labels = ["Never", "Up to 5%", "5% to 20%", "20% to 50%", "More than 50%"]
    # scl: 1 = never (pct==0), 2..5 = classes of the digitize on [~0,5,20,50]
    shares = [100 * (scl[valid] == i).mean() for i in range(1, 6)]
    (folder / "OAT_stability_summary.txt").write_text(
        "Share of valid pixels by share of the %d perturbed scenarios in which the Hazard class changes\n" % n_scen +
        "\n".join(f"  {l}: {v:.1f}%" for l, v in zip(labels, shares)) +
        f"\nBase thresholds (Multi-Otsu, 5 classes): {np.round(thr, 4).tolist()}\n", encoding="utf-8")
    pal = ["#9ECAE1", "#FFF3B0", "#FDB863", "#E66101", "#7F2704"]
    s = sites_df()
    outline = get_outline()
    ext = (b.left, b.right, b.bottom, b.top)
    extent = island_extent(outline, s) if outline is not None else ext
    ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])
    mapw = 11.6 * CM
    maph = mapw * ratio
    figh = maph + 5.4 * CM
    fig = plt.figure(figsize=(W_FULL, figh))
    ax = fig.add_axes([0.075, 1 - 0.04 - maph / figh, mapw / W_FULL, maph / figh])
    k = max(1, int(math.ceil(scl.shape[1] / 3200)))
    cm_ = ListedColormap(pal)
    ax.imshow(np.ma.masked_where(scl == 0, scl)[::k, ::k], extent=ext, origin="upper", cmap=cm_, vmin=0.5, vmax=5.5,
              interpolation="nearest", zorder=1)
    base_map(ax, outline, extent)
    ax.images[0].set_zorder(1)
    ax.scatter(s.x_utm36n, s.y_utm36n, s=12, c="white", edgecolors="black", linewidths=0.6, zorder=6)
    scalebar(ax, 50, north=True)
    panel_label(ax, "(a)")
    hs = [Patch(fc=pal[i], ec=EDGE, lw=0.4, label=labels[i]) for i in range(4, -1, -1)]
    hs.append(Line2D([0], [0], marker="o", lw=0, mfc="white", mec="black", ms=4.5, label="Site"))
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.075 + mapw / W_FULL + 0.015, 1 - 0.07), frameon=False, fontsize=6.4,
               title="Share of scenarios\nwith a class change", title_fontsize=6.4, alignment="left")
    axb = fig.add_axes([0.22, 0.10, 0.45, 0.17])
    axb.barh(range(5), shares[::-1], color=pal[::-1], edgecolor=EDGE, lw=0.5, height=0.7)
    for i, v in enumerate(shares[::-1]):
        axb.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=6.6)
    axb.set_yticks(range(5))
    axb.set_yticklabels(labels[::-1])
    axb.set_xlim(0, max(shares) * 1.18)
    axb.set_xlabel("Share of pixels (%)")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    panel_label(axb, "(b)", x=-0.38, y=1.05)
    save(fig, "Fig07_OAT_stability_map", copy_to=folder)
    write_readme(folder,
                 "FIG 7 - Stability of the Hazard class under the one-at-a-time weight perturbation (pixel level)\n\n"
                 "Rasters: OAT_stability_class.tif (1 never, 2 up to 5 pct, 3 5-20 pct, 4 20-50 pct, 5 more than 50 pct of the %d perturbed scenarios),\n"
                 "OAT_changes_percent.tif (continuous, percent of scenarios), OAT_changes_count.tif (number of scenarios).\n"
                 "Base class thresholds are kept fixed in every scenario (as in the pixel-level analysis of the paper).\n"
                 "Symbology of the class raster: unique values, colours #9ECAE1, #FFF3B0, #FDB863, #E66101, #7F2704\n"
                 "(scala blu-giallo-arancio-marrone: blu = stabile, marrone = instabile). Sites: white fill, black outline.\n"
                 "Class shares: OAT_stability_summary.txt.\n\n" % n_scen + PALETTE_TXT)
 
 
# ============================================================================ FIG 8: OAT, rank range
def fig8():
    set_style()
    p = oat_sites()
    p["name"] = p["name"].map(clean_name)
    p = p.sort_values(["R_class_base", "HI_rank_base"], ascending=[False, True]).iloc[::-1].reset_index(drop=True)
    n = len(p)
    fig, ax = plt.subplots(figsize=(W_FULL, (n * 0.33 + 2.6) * CM))
    for i, r in p.iterrows():
        c = CLASS3[int(r.R_class_base)]
        ax.plot([r.HI_rank_min - 0.35, r.HI_rank_max + 0.35], [i, i], color=EDGE, lw=6.2, solid_capstyle="butt", zorder=1.5)
        ax.plot([r.HI_rank_min - 0.3, r.HI_rank_max + 0.3], [i, i], color=c, lw=4.4, solid_capstyle="butt", zorder=2)
        ax.plot(r.HI_rank_base, i, marker="o", mfc="white", mec="#222222", ms=3.6, mew=0.8, zorder=3)
        if r.n_r_changes_fixed > 0:
            ax.plot(r.HI_rank_max + 0.9, i, marker="D", mfc="white", mec="#111111", ms=4.4, mew=1.1, zorder=4)
        ax.text(-0.9, i, f"R = {int(r.R_base)}", va="center", ha="left", fontsize=6.4, color="#444444")
    for i in range(n - 1):
        if p.R_class_base[i] != p.R_class_base[i + 1]:
            ax.axhline(i + 0.5, color="#999999", lw=0.7, ls="--", zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(p["name"], fontsize=6.4)
    ax.set_xlim(36, -5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 34])
    ax.set_xlabel("Hazard rank of the site (1 = highest Hazard Index); sites grouped by Risk class")
    ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, n - 0.2)
    hs = [Line2D([0], [0], color=CLASS3[k], lw=4.4, label=f"{LABEL3[k]} Risk") for k in (3, 2, 1)]
    hs += [Line2D([0], [0], marker="o", mfc="white", mec="#222222", lw=0, ms=3.6, mew=0.8, label="Rank in the main analysis"),
           Line2D([0], [0], marker="D", mfc="white", mec="#111111", lw=0, ms=4.4, mew=1.1,
                  label="Risk class changes in at least one\nscenario (original thresholds)")]
    ax.legend(handles=hs, loc="upper left", frameon=False, fontsize=6.4)
    fig.tight_layout()
    save(fig, "Fig08_OAT_rank_range")
 
 
# ============================================================================ FIG 9: remaining tests
OAT_RENAME = {"r_class_base": "R_class_base", "hi_rank_base": "HI_rank_base", "hi_rank_min": "HI_rank_min",
              "hi_rank_max": "HI_rank_max", "r_base": "R_base", "n_r_fixed_changes": "n_r_changes_fixed",
              "n_r_mot_changes": "n_r_changes_mot"}


def oat_sites():
    """Site-level OAT of script 05 (columns renamed for the plotting code)."""
    return pd.read_csv(result("sensitivity", "oat_sites_per_site.csv")).rename(columns=OAT_RENAME)


def risk_tests():
    """Risk class (1-3) of each site under the single-run tests of script 05."""
    s = sites_df()
    v = pd.read_csv(result("sensitivity", "vulnerability_tests_sites.csv")).set_index("id").loc[s.id]
    c = pd.read_csv(result("sensitivity", "coastal_tests_sites.csv")).set_index("id").loc[s.id]
    a = pd.read_csv(result("sensitivity", "aggregation_sites.csv")).set_index("id").loc[s.id]
    out = pd.DataFrame({"id": s.id, "name": s["name"]})
    out["Main analysis"] = s.Rc.values
    out["No Enclosure, Shelter"] = v["no_enclosure_shelter"].values
    for k in (1, 2, 3):
        out[f"Fixed thresholds {k}"] = v[f"fixed_thresholds_{k}"].values
    out["Without CPI"] = c["r_class_no_cpi"].values
    out["Without CPI and\nDtC"] = c["r_class_no_cpi_dtc"].values
    out["Without CPI, DtC and Elevation"] = c["r_class_no_cpi_dtc_elevation"].values
    out["_top_product"] = a["top_product"].values
    out["_top_meanrank"] = a["top_mean_rank"].values
    return out, {"poly": False, "nocpi": True}


def fig9():
    set_style()
    s = sites_df()
    tab, avail = risk_tests()
    oat = oat_sites().set_index("id").loc[s.id]
    tab = tab.set_index("id").loc[s.id].reset_index()
    order = s.sort_values(["R", "HI"], ascending=[False, False]).index
    tab = tab.loc[order].reset_index(drop=True)
    oat = oat.loc[s.loc[order, "id"].values].reset_index()
    tcols = [c for c in tab.columns if c not in ("id", "name") and not c.startswith("_")]
    n, nt = len(tab), len(tcols)
    rowh = 0.33
    fig = plt.figure(figsize=(W_FULL, (n * rowh + 6.0) * CM))
    ax = fig.add_axes([0.285, 1.6 / (n * rowh + 6.0), 0.69, (n * rowh + 3.9) / (n * rowh + 6.0)])
    ncols_total = nt + 1.0 + 2 + 1.0 + (2 if INCLUDE_MEAN_RANK else 1)
    ax.set_xlim(-0.2, ncols_total + 0.6)
    ax.set_ylim(-2.8, n + 6.6)
    ax.axis("off")
    letters = {1: "L", 2: "M", 3: "H"}
    base = tab["Main analysis"].values
    # block A
    for j, c in enumerate(tcols):
        ax.text(j + 0.55, n - 0.1, c.replace("\n", " "), rotation=60, ha="left", va="bottom", fontsize=6.0)
        for i in range(n):
            v = int(tab.loc[i, c])
            ax.add_patch(Rectangle((j, n - 1 - i - 0.42), 0.9, 0.84, fc=CLASS3[v], ec="white", lw=0.5))
            ax.text(j + 0.45, n - 1 - i, letters[v], ha="center", va="center", fontsize=5.8, color=text_on(CLASS3[v]))
        chg = int((tab[c].values != base).sum())
        hi = int((tab[c].values == 3).sum())
        ax.text(j + 0.45, -1.1, "–" if c == "Main analysis" else str(chg), ha="center", va="center", fontsize=6.2)
        ax.text(j + 0.45, -2.1, str(hi), ha="center", va="center", fontsize=6.2)
    ax.text(-0.3, -1.1, "Sites changing class", ha="right", va="center", fontsize=6.4)
    ax.text(-0.3, -2.1, "Sites in the High class", ha="right", va="center", fontsize=6.4)
    # block B: weights
    x0 = nt + 0.8
    for k, (col, lab) in enumerate((("n_r_changes_fixed", "Weights, original thr."),
                                    ("n_r_changes_mot", "Weights, new thr."))):
        ax.text(x0 + k + 0.55, n - 0.1, lab.replace("\n", " "), rotation=60, ha="left", va="bottom", fontsize=6.0)
        nscen = int(oat["n_scenarios"].iloc[0])
        for i in range(n):
            v = int(oat.loc[i, col])
            shade = 1 - 0.55 * min(v / 16, 1)
            ax.add_patch(Rectangle((x0 + k, n - 1 - i - 0.42), 0.9, 0.84, fc=(shade, shade, shade) if v else "#F7F7F7",
                                   ec="white", lw=0.5))
            ax.text(x0 + k + 0.45, n - 1 - i, str(v) if v else "·", ha="center", va="center", fontsize=5.8,
                    color="black" if v else "#999999")
    # block C: five highest by aggregation
    x1 = x0 + 2.8
    cc = [("_top_product", "Five highest, product")]
    if INCLUDE_MEAN_RANK:
        cc.append(("_top_meanrank", "Five highest, mean rank"))
    for k, (col, lab) in enumerate(cc):
        ax.text(x1 + k + 0.55, n - 0.1, lab.replace("\n", " "), rotation=60, ha="left", va="bottom", fontsize=6.0)
        for i in range(n):
            ax.add_patch(Rectangle((x1 + k, n - 1 - i - 0.42), 0.9, 0.84, fc="#F7F7F7", ec="white", lw=0.5))
            if tab.loc[i, col]:
                ax.scatter([x1 + k + 0.45], [n - 1 - i], s=16, c="black", zorder=5)
    for i in range(n):
        ax.text(-0.3, n - 1 - i, short_name(tab.loc[i, "name"]), ha="right", va="center", fontsize=6.2)
    for i in range(n - 1):
        if s.loc[order[i], "Rc"] != s.loc[order[i + 1], "Rc"]:
            ax.plot([-0.1, ncols_total], [n - 1 - i - 0.5] * 2, color=GREY, lw=0.6, ls="--")
    h3 = [Patch(fc=CLASS3[i], ec=EDGE, lw=0.4, label=f"{letters[i]}  {LABEL3[i]} Risk") for i in range(1, 4)]
    h3 += [Patch(fc="#BDBDBD", ec=EDGE, lw=0.4, label="Weight scenarios with a Risk class change (of 32); thr. = thresholds"),
           Line2D([0], [0], marker="o", lw=0, mfc="black", mec="black", ms=4, label="Among the five highest values")]
    fig.legend(handles=h3, loc="lower left", bbox_to_anchor=(0.05, 0.003), ncol=3, frameon=False, fontsize=6.4,
               columnspacing=1.2, handletextpad=0.5)
    save(fig, "Fig09_Remaining_tests")


# ============================================================================ EXTENDED DATA: eight factors
def ext_factors():
    set_style()
    import rasterio
    WEIGHTS = load_weights()
    paths = {c: CFG.input("factors", c) for c in CFG.factor_codes}
    folder = MAPS_DIR / "ExtData_factors"
    write_sites_shp(folder)
    copy_outline(folder)
    s = sites_df()
    outline = get_outline()
    names = list(paths)
    first, ext, _, k = read_raster(paths[names[0]])
    extent = island_extent(outline, s) if outline is not None else ext
    ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])
    pw_cm = 7.3
    ph_cm = pw_cm * ratio
    top_cm, gap_cm, bottom_cm = 0.55, 0.6, 1.6   # title room above row 1, room between rows, colour bar margin
    figh_cm = top_cm + 4 * ph_cm + 3 * gap_cm + bottom_cm
    figh = figh_cm * CM
    fig = plt.figure(figsize=(W_FULL, figh))
    cmap = LinearSegmentedColormap.from_list("hazard", [CLASS5[i] for i in range(1, 6)])
    cmap.set_bad((0, 0, 0, 0))
    ims = []
    for i, nme in enumerate(names):
        a, e_, _, k = read_raster(paths[nme])
        r, c = divmod(i, 2)
        x = 0.065 + c * 0.47
        y_cm = figh_cm - top_cm - (r + 1) * ph_cm - r * gap_cm
        ax = fig.add_axes([x, y_cm / figh_cm, pw_cm * CM / W_FULL, ph_cm / figh_cm])
        im = ax.imshow(np.ma.masked_invalid(a[::k, ::k]), extent=e_, origin="upper", cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest", zorder=1)
        base_map(ax, outline, extent)
        ax.images[0].set_zorder(1)
        ax.tick_params(labelsize=5, labelbottom=(r == 3), labelleft=(c == 0))
        ax.scatter(s.x_utm36n, s.y_utm36n, s=7, c="black", edgecolors="white", linewidths=0.4, zorder=6)
        code = nme
        ax.set_title(f"({chr(97 + i)}) {code}, weight {WEIGHTS[code]:.3f}", fontsize=7.4, loc="left", pad=3)
        ims.append(im)
    cax = fig.add_axes([0.25, 0.75 / figh_cm, 0.5, 0.012])
    cb = fig.colorbar(ims[0], cax=cax, orientation="horizontal", ticks=np.linspace(0, 1, 6))
    cb.set_label("Normalised factor value", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    save(fig, "ExtData_Hazard_factors", copy_to=folder)
    lst = "\n".join(f"  {n}: {paths[n].name}" for n in names)
    write_readme(folder,
                 "EXTENDED DATA - the eight normalised Hazard factors (before propagation)\n\n"
                 "Rasters (already normalised to [0, 1], 30 m):\n" + lst + "\n"
                 "Symbology: the same continuous ramp YlGnBu, stretch fixed 0 to 1, no data transparent, for all eight panels.\n"
                 "Layout: 2 columns x 4 rows, panel title '(a) SAI (weight 0.131)', shared colour bar at the bottom.\n"
                 "Sites: small black dots (Sites_HVER.shp). Same extent and outline in all panels.\n\n" + PALETTE_TXT)
 
 
# ============================================================================ COMMAND LINE
FIGS = {
    "fig2": (fig2, "workflow diagram"),
    "fig3": (fig3, "cascade weights: network and weights"),
    "fig4": (fig4, "Hazard Index, continuous and classified (map preview)"),
    "fig5": (fig5, "Vulnerability, Exposure, Risk maps + Paphos-Limassol detail (map preview)"),
    "fig6": (fig6, "ranking matrix of the 34 sites"),
    "fig7": (fig7, "OAT stability map, pixel level (map preview)"),
    "fig8": (fig8, "OAT, Hazard rank range of the sites"),
    "fig9": (fig9, "Risk class under the sensitivity tests"),
    "ext": (ext_factors, "eight normalized Hazard factors (Extended data Figure 1)"),
}
 
 
def make(name):
    key = name.lower().replace("figure", "fig").replace(" ", "")
    if key.isdigit():
        key = "fig" + key
    if key not in FIGS:
        raise KeyError(f"unknown figure '{name}'. Choose one of: {', '.join(FIGS)}")
    print(f"Building {key}: {FIGS[key][1]}")
    FIGS[key][0]()
 
 
def main():
    ap = argparse.ArgumentParser(description="COASTLINE publication figures (default: all)")
    ap.add_argument("figure", nargs="?", help="fig2 ... fig9, ext, all, list")
    a = ap.parse_args()
    choice = a.figure
    if choice is None:
        choice = "all"
    if choice == "list":
        for k, (_, d) in FIGS.items():
            print(f"  {k:5s} {d}")
        return
    if choice == "all":
        for k in FIGS:
            try:
                make(k)
            except Exception as exc:
                print(f"  {k} skipped: {type(exc).__name__}: {exc}")
        return
    make(choice)
 
 
if __name__ == "__main__":
    main()