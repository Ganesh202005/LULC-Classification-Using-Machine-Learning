# V2.0 — Integration Guide

This is not a full new project copy — it's the five files that needed to
change, plus four matching notebooks. Drop them into your existing
`D:\claude\project` tree alongside what you already have.

## Where each file goes

```
D:\claude\project\
├── scripts\
│   ├── config.py              <- leave as-is (or delete, your choice)
│   ├── preprocessing.py       <- leave as-is
│   ├── training.py            <- leave as-is
│   ├── classification.py      <- leave as-is
│   ├── raster_io_v2.py        <- ADD (new file)
│   ├── config_v2.py           <- ADD (new file)
│   ├── preprocessing_v2.py    <- ADD (new file)
│   ├── training_v2.py         <- ADD (new file)
│   └── classification_v2.py   <- ADD (new file)
├── notebooks\
│   ├── 01_data_preparation.ipynb        <- leave as-is
│   ├── ...
│   ├── 01_data_preparation_v2.ipynb     <- ADD
│   ├── 02_extract_training_data_v2.ipynb <- ADD
│   ├── 03_train_models_v2.ipynb         <- ADD
│   └── 04_classify_raster_v2.ipynb      <- ADD
├── qgis\
│   ├── training.gpkg           <- already there, unchanged, now used directly
│   └── mumbai_lulc.gpkg        <- already there, NOT used by V2 (see below)
├── data\, models\, outputs\    <- already there, unchanged
```

Nothing needs to be renamed or moved. V1 files are untouched; V2 files are
additive and self-contained (they import `config_v2`, not `config`).

## Why mumbai_lulc.gpkg isn't used

Both `training.gpkg` and `mumbai_lulc.gpkg` were uploaded. Only
`training.gpkg` is used by `config_v2.py`'s default. Concretely:

| | `training.gpkg` (used) | `mumbai_lulc.gpkg` (not used) |
|---|---|---|
| Layer | `training_polygons` | `mumbai_lulc` |
| Polygons | 28 | 32 |
| Classes | 1=water, 2=built_up, 3=vegetation, 4=agriculture, 5=barren | 1=water, 2=built_up, 3=vegetation, 4=barren (**no agriculture**) |
| `class_name` field | `class_name` | `class_name ` (trailing space) |

`training.gpkg` has the full 5-class scheme matching your project's
intended classes, so it's the default. If you actually intended
`mumbai_lulc.gpkg` to be the training set, change one line in
`config_v2.py`:

```python
TRAINING_POLYGONS_PATH = QGIS_DIR / "mumbai_lulc.gpkg"
TRAINING_POLYGONS_LAYER = "mumbai_lulc"
```

`training_v2.py` will then report its 4 actual classes at runtime and
work correctly with them — it does not require exactly 5 classes.

`training_NDBI.csv` isn't used either: it's a QGIS Zonal Statistics
export of `mumbai_lulc.gpkg`'s 32 polygons (one row per polygon, columns
like `B02_mean`, `NDVI_mean`; no B12 at all). The pipeline needs one row
per *pixel*, so this file's granularity doesn't match what
`training_v2.py` extracts — it isn't something the code could consume
even after fixing the crash.

## What to actually do

1. Copy the 5 new `scripts/*_v2.py` files and 4 new `notebooks/*_v2.ipynb`
   files into your existing project.
2. Open `scripts/config_v2.py` and check `RAW_BANDS` points at your real
   downloaded filenames (or just leave them — V2 auto-detects by band
   code if the exact name doesn't match).
3. Run the four `*_v2` notebooks in order: 01 → 02 → 03 → 04.
4. `outputs/classified_map.tif` is your final result — open it in QGIS.

See the main chat response for the full root-cause diagnosis and expected
console output at each step.
