"""Names, labels and colours shared by the whole framework."""

# Hazard factors (Table 3 of the article), in the order used everywhere.
FACTOR_NAMES = {
    "SAI": "Seismic Activity",
    "BUI": "Built-up Density",
    "LSI": "Landslide Susceptibility",
    "CPI": "Coastal Proximity",
    "FFI": "Fire Footprint",
    "RNI": "Road Proximity",
    "SLRI": "Soil Loss Rate",
    "FHI": "Fluvial Hydrography",
}

# Class labels
LABELS_3 = {1: "Low", 2: "Medium", 3: "High"}
LABELS_5 = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}

# Vulnerability attributes (Table 4): column of the score and direction.
# inverted=True -> a LOWER value is MORE vulnerable (score 3).
CATEGORICAL_ATTRIBUTES = ["heritage", "enclosure", "shelter", "land_use"]
CONTINUOUS_ATTRIBUTES = {
    "elevation_m": True,        # low-lying land is reached first by the sea
    "dist_coast_m": True,       # close to the shoreline
    "slope_deg": False,         # steep slopes erode faster
}
SCORE_COLUMNS = [f"s_{a}" for a in CATEGORICAL_ATTRIBUTES + list(CONTINUOUS_ATTRIBUTES)]

# ESA WorldCover classes (value 0 = no data)
WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare / sparse vegetation", 70: "Snow and ice", 80: "Permanent water bodies",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen",
}

# Publication palette: one green-yellow-red scale for every figure.
CLASS3_COLORS = {1: "#1B9E77", 2: "#F2C12E", 3: "#D62F25"}
CLASS5_COLORS = {1: "#1B9E77", 2: "#A6D96A", 3: "#F2C12E", 4: "#F58A2E", 5: "#D62F25"}
