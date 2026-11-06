"""
classification_v2.py
=====================
V2.0 changes from classification.py:

  - final_stack.tif's nodata pixels are NaN (float32) in V2, so the
    valid-pixel mask here works directly off that instead of an
    assumed sentinel value.
  - The output classified raster is written via
    raster_io_v2.write_geotiff(), consistent with the rest of V2, and
    explicitly gets nodata=0 with dtype uint8 -- confirmed to open
    correctly in QGIS (see the smoke test in the accompanying README).
"""

import numpy as np
import pandas as pd
import rasterio

import config_v2 as config
import raster_io_v2 as rio


def load_raster_for_prediction(raster_path=None):
    """Load the 9-band feature stack and reshape to (pixels, bands)."""
    raster_path = raster_path or config.FINAL_STACK_PATH

    with rasterio.open(raster_path) as src:
        image = src.read()
        profile = src.profile.copy()

    bands, rows, cols = image.shape
    print(f"Loaded raster: {bands} bands, {rows} rows, {cols} cols ({rows * cols:,} pixels)")

    X_image = image.reshape(bands, rows * cols).T
    return X_image, rows, cols, profile


def predict_full_raster(model, X_image, rows, cols, nodata_value=0, batch_size=500_000):
    """Predict every valid (non-NaN) pixel; nodata pixels stay
    nodata_value in the output."""
    n_pixels = X_image.shape[0]
    predictions = np.full(n_pixels, nodata_value, dtype="uint8")

    valid_mask = ~np.isnan(X_image).any(axis=1)
    valid_idx = np.where(valid_mask)[0]

    print(f"Predicting {len(valid_idx):,} valid pixels out of {n_pixels:,} total "
          f"({n_pixels - len(valid_idx):,} are nodata)...")

    for start in range(0, len(valid_idx), batch_size):
        end = min(start + batch_size, len(valid_idx))
        chunk_idx = valid_idx[start:end]
        chunk_df = pd.DataFrame(X_image[chunk_idx], columns=config.FEATURE_NAMES)
        chunk_pred = model.predict(chunk_df)
        predictions[chunk_idx] = chunk_pred
        print(f"  predicted pixels {start:,}-{end:,} / {len(valid_idx):,}")

    return predictions.reshape(rows, cols)


def save_classified_geotiff(classified: np.ndarray, profile: dict, out_path=None) -> str:
    out_path = out_path or config.CLASSIFIED_MAP_PATH
    return rio.write_geotiff(out_path, classified.astype("uint8"), profile, nodata=0, dtype="uint8")


def classify_full_scene(model, raster_path=None, out_path=None) -> str:
    X_image, rows, cols, profile = load_raster_for_prediction(raster_path)
    classified = predict_full_raster(model, X_image, rows, cols)
    path = save_classified_geotiff(classified, profile, out_path)
    info = rio.inspect_raster(path)
    print(f"\nclassified_map.tif: driver={info['driver']}, dtype={info['dtype']}, "
          f"CRS={info['crs']}, nodata={info['nodata']}")
    return path
