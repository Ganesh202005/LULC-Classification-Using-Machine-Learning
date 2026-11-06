"""
raster_io_v2.py
================
NEW in V2.0.

This module exists because of the root cause of the pipeline failure:

    CPLE_NotSupportedError: JP2 driver only supports creating Byte,
    GDT_Int16, GDT_UInt16, GDT_Int32, GDT_UInt32

WHY THIS HAPPENED (confirmed by reproducing it):
Every write function in the V1 pipeline built its output metadata with
`out_meta = src.meta.copy()`. When the *source* being read was a raw
Sentinel-2 `.jp2` file, `src.meta['driver']` is `'JP2OpenJPEG'`. That
driver name was never overridden -- it was carried, unchanged, through
every downstream write:

    raw .jp2 (JP2OpenJPEG, uint16)
      -> clipped "*_clip.tif"      (still driver=JP2OpenJPEG, uint16 -- WORKS,
                                     because JP2 supports UInt16, even though
                                     the file is named ".tif")
      -> resampled band            (still JP2OpenJPEG, uint16 -- WORKS)
      -> sentinel_stack.tif        (still JP2OpenJPEG, uint16 -- WORKS)
      -> ndvi.tif (float32)        (still JP2OpenJPEG -- FAILS: JP2 cannot
                                     write float32)

Because the crash happens inside compute_indices(), build_final_stack()
never runs, so final_stack.tif is never created -- which is exactly the
second error you saw ("final_stack.tif does not exist").

THE FIX:
Every single raster write in this project now goes through
`write_geotiff()` below, which unconditionally sets driver="GTiff"
regardless of what format was read. It is impossible for this specific
bug to recur as long as every write call in preprocessing_v2.py,
training_v2.py and classification_v2.py uses this function (they all do).
"""

from pathlib import Path

import numpy as np
import rasterio


# GDAL creation options applied to every GeoTIFF this project writes.
# LZW compression + tiling keeps intermediate files small and fast to
# read back for the ML steps.
GTIFF_CREATION_OPTS = {
    "compress": "lzw",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
    "BIGTIFF": "IF_SAFER",
}


def write_geotiff(path, data: np.ndarray, meta: dict, nodata=None, dtype=None) -> str:
    """Write a numpy array to a GeoTIFF -- always via the GTiff driver.

    This is the ONE place in the whole project that writes a raster to
    disk for permanent pipeline outputs. Every caller in
    preprocessing_v2.py / training_v2.py / classification_v2.py routes
    through here specifically so the JP2-driver-inheritance bug (see
    module docstring) cannot come back.

    Parameters
    ----------
    path : path-like
        Output file path. Parent directories are created if missing.
    data : np.ndarray
        2D (single band, shape (H, W)) or 3D (multi-band,
        shape (bands, H, W)) array.
    meta : dict
        A rasterio meta/profile dict to base the output on (typically
        `src.meta.copy()` from an input raster). Its `driver`, `count`,
        `height`, `width` and `dtype` are OVERWRITTEN here based on
        `data` -- you do not need to (and should not try to) set those
        yourself before calling this.
    nodata : number, optional
        NoData value to stamp into the output. If omitted, whatever is
        already in `meta['nodata']` is kept (which may be None).
    dtype : str or numpy dtype, optional
        Output dtype. Defaults to `data.dtype`.

    Returns
    -------
    str : the path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    elif data.ndim != 3:
        raise ValueError(f"write_geotiff expects a 2D or 3D array, got shape {data.shape}")

    out_dtype = dtype or data.dtype

    out_meta = dict(meta)  # copy, don't mutate caller's dict
    out_meta.update(
        driver="GTiff",  # <-- THE FIX: always GTiff, never inherited
        count=data.shape[0],
        height=data.shape[1],
        width=data.shape[2],
        dtype=out_dtype,
    )
    out_meta.update(GTIFF_CREATION_OPTS)
    if nodata is not None:
        out_meta["nodata"] = nodata

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(data.astype(out_dtype))

    return str(path)


def inspect_raster(path) -> dict:
    """Read and return a raster's key metadata without loading pixel
    data. Used throughout V2 to validate inputs *before* processing them
    (requirement: "validate input files before processing").
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "driver": src.driver,
            "dtype": src.dtypes[0],
            "count": src.count,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "resolution": src.res,
            "nodata": src.nodata,
            "bounds": tuple(src.bounds),
        }


def find_band_file(raw_dir, band_code: str, configured_path=None) -> Path:
    """Locate the raw file for one Sentinel-2 band, tolerant of naming
    differences from what config.py assumed.

    Resolution order:
      1. `configured_path` from config.py, if it actually exists.
      2. Any single file directly inside `raw_dir` whose name contains
         `band_code` (case-insensitive) and has a raster-like extension
         (.jp2 / .tif / .tiff / .img). Handles cases like the configured
         filename being slightly wrong (different acquisition timestamp,
         tile ID, or a "_10m"/"_20m" suffix you didn't include).

    Raises FileNotFoundError with the *actual* directory contents if
    nothing can be matched, rather than guessing or inventing a path.
    Raises FileNotFoundError if MULTIPLE files match ambiguously, asking
    you to set the path explicitly rather than picking one silently.
    """
    if configured_path is not None and Path(configured_path).exists():
        return Path(configured_path)

    raw_dir = Path(raw_dir)
    raster_exts = {".jp2", ".tif", ".tiff", ".img"}

    candidates = []
    if raw_dir.exists():
        candidates = sorted(
            p
            for p in raw_dir.iterdir()
            if p.is_file()
            and band_code.lower() in p.name.lower()
            and p.suffix.lower() in raster_exts
        )

    if len(candidates) == 1:
        print(f"  (auto-detected {band_code} -> {candidates[0].name}; "
              f"configured path did not exist)")
        return candidates[0]

    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Ambiguous: multiple files in {raw_dir} match band '{band_code}': "
            f"{[c.name for c in candidates]}. "
            f"Set RAW_BANDS['{band_code}'] in config_v2.py explicitly."
        )

    existing = sorted(p.name for p in raw_dir.iterdir()) if raw_dir.exists() else []
    raise FileNotFoundError(
        f"Could not find a raw file for band '{band_code}'.\n"
        f"  Configured path checked: {configured_path}\n"
        f"  Files actually present in {raw_dir}: {existing or '(directory is empty or missing)'}"
    )


def apply_reflectance_scaling(data: np.ndarray, scale: float, offset: float = 0.0) -> np.ndarray:
    """Convert raw Sentinel-2 L2A digital numbers to surface reflectance
    (approximately 0-1 range): (DN + offset) / scale.

    - `scale` is the standard Sentinel-2 L2A reflectance scale factor
      (10000 by default -- ESA's QUANTIFICATION_VALUE).
    - `offset` should stay 0 unless you have confirmed your product's
      metadata (MTD_MSIL2A.xml -> BOA_ADD_OFFSET) uses the post-2022
      processing-baseline offset (-1000 in the DN, so pass +1000 here
      to cancel it out before scaling). We do NOT enable this by
      default because we cannot verify it from a vector file or CSV --
      only from your actual product metadata, which was not uploaded.
    """
    return (data.astype("float32") + offset) / float(scale)
