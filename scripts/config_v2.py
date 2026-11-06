"""
config_v2.py
============
Single source of truth for every path and setting.

WHAT CHANGED FROM V1's config.py, AND WHY -- based on inspecting your
actual uploaded files, not on assumptions:

1. CLASSES was missing class 5 ("Barren"). Your uploaded
   `training.gpkg` (layer "training_polygons") actually contains 5
   classes: 1=water, 2=built_up, 3=vegetation, 4=agriculture,
   5=barren -- 28 polygons total. This file now matches that.

2. AOI_PATH pointed at "mumbai_aoi.gpkg", which was never uploaded and
   does not exist in your project. You do NOT need to go back to QGIS
   for this: preprocessing_v2.py now auto-derives a bounding-box AOI
   from your training polygons (with a buffer) if no AOI file is
   present, and tells you clearly when it does this. If you DO have a
   real AOI polygon, just put it at AOI_PATH and it will be used
   instead -- see AOI_PATH below.

3. TRAINING_POLYGONS_PATH now points at "training.gpkg" (the file you
   actually uploaded), not the old placeholder name
   "training_polygons.gpkg".

4. IMPORTANT -- you uploaded a SECOND polygon file, "mumbai_lulc.gpkg",
   which is NOT used by this config. Here's exactly why:
     - It has only 4 classes (1=water, 2=built_up, 3=vegetation,
       4=barren) -- "agriculture" is missing entirely, unlike
       training.gpkg's 5-class scheme.
     - Its class_name column is literally named "class_name " (with a
       trailing space) -- a QGIS export quirk. training_v2.py's polygon
       loader strips whitespace from column names automatically now,
       so this would not crash, but the missing class is a real
       difference, not a formatting issue.
     - You also uploaded "training_NDBI.csv", which turns out to be a
       QGIS Zonal-Statistics export of mumbai_lulc.gpkg's 32 polygons
       (one row per POLYGON, not per pixel -- columns are B02_mean,
       B03_mean, ..., NDVI_mean, NDWImean, NDBImean). It has no B12 at
       all and doesn't match the per-pixel training format this
       pipeline needs, so it cannot be fed to training_v2.py directly.
   If mumbai_lulc.gpkg is actually the training set you intended (not
   training.gpkg), change TRAINING_POLYGONS_PATH below to point at it
   -- the code will adapt to its 4 classes automatically and will warn
   you about the mismatch with the CLASSES dict rather than failing.

All paths are pathlib-based and relative to PROJECT_ROOT, so this works
unchanged whether the project lives at D:\\claude\\project or anywhere
else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 2. DATA FOLDERS
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "sentinel2"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
QGIS_DIR = PROJECT_ROOT / "qgis"

for _d in [RAW_DIR, PROCESSED_DIR, TRAINING_DIR, MODELS_DIR, OUTPUTS_DIR, QGIS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 3. AOI (Area of Interest) -- OPTIONAL in V2
# ---------------------------------------------------------------------------
# If this file exists, it is used to clip the Sentinel-2 scene.
# If it does NOT exist (your current situation), preprocessing_v2.py
# automatically derives a rectangular AOI from the union of your
# training polygons' bounds, expanded by AOI_BUFFER_METERS, and saves
# it to qgis/auto_generated_aoi.gpkg so you can inspect it in QGIS.
AOI_PATH = QGIS_DIR / "mumbai_aoi.gpkg"
AOI_LAYER = None
AOI_BUFFER_METERS = 300  # only used when AOI_PATH doesn't exist

# ---------------------------------------------------------------------------
# 4. SENTINEL-2 BANDS
# ---------------------------------------------------------------------------
# Put your downloaded Sentinel-2 L2A band files in RAW_DIR. Exact
# filenames below are a best guess; if they don't match what you
# actually downloaded, raster_io_v2.find_band_file() will look for any
# file in RAW_DIR whose name contains the band code (e.g. "B02") and
# use that instead, so you don't have to rename anything by hand.
RAW_BANDS = {
    "B02": RAW_DIR / "T43QBB_20260106T054231_B02.jp2",
    "B03": RAW_DIR / "T43QBB_20260106T054231_B03.jp2",
    "B04": RAW_DIR / "T43QBB_20260106T054231_B04.jp2",
    "B08": RAW_DIR / "T43QBB_20260106T054231_B08.jp2",
    "B11": RAW_DIR / "T43QBB_20260106T054231_B11.jp2",
    "B12": RAW_DIR / "T43QBB_20260106T054231_B12.jp2",
}

TARGET_RESOLUTION = 10
BANDS_TO_RESAMPLE = ["B11", "B12"]
RESAMPLING_METHOD = "bilinear"
BAND_ORDER = ["B02", "B03", "B04", "B08", "B11", "B12"]

# ---------------------------------------------------------------------------
# 4b. RADIOMETRIC SCALING
# ---------------------------------------------------------------------------
# Sentinel-2 L2A bands are delivered as scaled integer digital numbers
# (DN), not reflectance. Converting DN -> reflectance (roughly 0-1)
# makes NDVI/NDWI/NDBI numerically identical (they're ratios, scale
# cancels out) but makes the raw B02..B12 feature values interpretable
# and keeps everything in a consistent float32 range for the ML models.
APPLY_REFLECTANCE_SCALING = True
REFLECTANCE_SCALE = 10000.0   # ESA's standard L2A QUANTIFICATION_VALUE

# Sentinel-2 processing baseline >= 04.00 (used for products generated
# after ~25 Jan 2022) applies a -1000 DN offset before scaling. We do
# NOT enable this correction by default because we cannot confirm your
# product's baseline from the files you uploaded (only visible in your
# SAFE product's MTD_MSIL2A.xml, which wasn't provided). If your NDVI
# values look implausibly shifted, check that XML for BOA_ADD_OFFSET
# and set this to +1000 if it's present.
RADIOMETRIC_OFFSET = 0.0

# ---------------------------------------------------------------------------
# 5. INTERMEDIATE / OUTPUT RASTER PATHS
# ---------------------------------------------------------------------------
CLIPPED_BANDS = {b: PROCESSED_DIR / f"{b}_clip.tif" for b in BAND_ORDER}
STACK_PATH = PROCESSED_DIR / "sentinel_stack.tif"
NDVI_PATH = PROCESSED_DIR / "ndvi.tif"
NDWI_PATH = PROCESSED_DIR / "ndwi.tif"
NDBI_PATH = PROCESSED_DIR / "ndbi.tif"
FINAL_STACK_PATH = PROCESSED_DIR / "final_stack.tif"

# ---------------------------------------------------------------------------
# 6. FEATURES
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDWI", "NDBI"]

FEATURE_SETS = {
    "bands_only": ["B02", "B03", "B04", "B08", "B11", "B12"],
    "indices_only": ["NDVI", "NDWI", "NDBI"],
    "all_features": FEATURE_NAMES,
}

# ---------------------------------------------------------------------------
# 7. TRAINING POLYGONS
# ---------------------------------------------------------------------------
# Defaults to your uploaded "training.gpkg" (5-class scheme, matches
# CLASSES below). See the module docstring for why mumbai_lulc.gpkg is
# NOT the default. To use it instead, change the line below.
TRAINING_POLYGONS_PATH = QGIS_DIR / "training.gpkg"
TRAINING_POLYGONS_LAYER = "training_polygons"

# Column names are matched case-insensitively with whitespace stripped
# by training_v2.py, so "class_name " (as in mumbai_lulc.gpkg) will
# still be found even though it's written here without the space.
CLASS_ID_FIELD = "class_id"
CLASS_NAME_FIELD = "class_name"

# Informational display names, used for confusion-matrix axis labels
# and the final map legend. training_v2.py cross-checks this against
# the class_id/class_name values ACTUALLY found in your polygon layer
# at runtime and will warn (not fail) on any mismatch -- it never
# blindly trusts this dict over your real data.
CLASSES = {
    1: "Water",
    2: "Built-up",
    3: "Vegetation",
    4: "Agriculture",
    5: "Barren",
}

# ---------------------------------------------------------------------------
# 8. EXTRACTED TRAINING DATA
# ---------------------------------------------------------------------------
TRAINING_CSV_PATH = TRAINING_DIR / "training_pixels.csv"

# ---------------------------------------------------------------------------
# 9. TRAIN / TEST SPLIT + MODEL SETTINGS
# ---------------------------------------------------------------------------
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Cap on training pixels per class before model fitting -- see
# training_v2.subsample_training_data() for why this exists (spatial
# autocorrelation + SVM training time). Set to None to use every
# extracted pixel.
MAX_SAMPLES_PER_CLASS = 3000

RF_PARAMS = dict(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
SVM_PARAMS = dict(kernel="rbf")
LR_PARAMS = dict(max_iter=1000)

# ---------------------------------------------------------------------------
# 10. MODEL / OUTPUT ARTIFACT PATHS
# ---------------------------------------------------------------------------
RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
SVM_MODEL_PATH = MODELS_DIR / "svm.joblib"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"

MODEL_COMPARISON_CSV = OUTPUTS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PNG = OUTPUTS_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PNG = OUTPUTS_DIR / "feature_importance.png"
FEATURE_IMPORTANCE_CSV = OUTPUTS_DIR / "feature_importance.csv"

CLASSIFIED_MAP_PATH = OUTPUTS_DIR / "classified_map.tif"

FINAL_MODEL_FOR_PREDICTION = "random_forest"  # random_forest / svm / logistic_regression
