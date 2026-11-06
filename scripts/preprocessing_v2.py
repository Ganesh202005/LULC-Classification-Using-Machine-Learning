"""
preprocessing_v2.py
====================
V2.0 of the Sentinel-2 preprocessing pipeline. Same conceptual steps as
before (clip -> resample -> align -> stack -> indices -> final stack),
rewritten so that:

  1. Every raster write goes through raster_io_v2.write_geotiff(), which
     always forces driver="GTiff". This is the fix for the
     CPLE_NotSupportedError root cause -- see raster_io_v2.py's
     module docstring for the full explanation and reproduction.

  2. Raw bands are converted to float32 reflectance immediately after
     clipping (config.APPLY_REFLECTANCE_SCALING), so every intermediate
     file from that point on is float32 GeoTIFF -- consistent dtype,
     no more "does this format support this dtype" questions anywhere
     downstream.

  3. An AOI is not mandatory. If config.AOI_PATH doesn't exist, an AOI
     is auto-derived from your training polygons' bounds (see
     get_or_build_aoi()) instead of stopping and asking you to draw one
     in QGIS.

  4. Every raw band lookup goes through raster_io_v2.find_band_file(),
     which tolerates the actual downloaded filename differing slightly
     from what config.py assumed.

  5. Inputs are validated and reported clearly (inspect_all_raw_bands())
     before any processing starts.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.warp import reproject

import config_v2 as config
import raster_io_v2 as rio


# ---------------------------------------------------------------------------
# Validation: inspect every raw band before touching anything
# ---------------------------------------------------------------------------
def resolve_raw_band_paths() -> dict:
    """Resolve the actual on-disk path for every band in config.RAW_BANDS,
    using raster_io_v2.find_band_file() as a fallback. Returns
    {band_code: Path}. Raises FileNotFoundError with a clear message
    (naming exactly which band and what IS present) if any band is
    genuinely missing.
    """
    resolved = {}
    errors = []
    for band_key, configured_path in config.RAW_BANDS.items():
        try:
            resolved[band_key] = rio.find_band_file(config.RAW_DIR, band_key, configured_path)
        except FileNotFoundError as e:
            errors.append(str(e))
    if errors:
        raise FileNotFoundError(
            "Cannot proceed -- missing raw Sentinel-2 band file(s):\n\n"
            + "\n\n".join(errors)
        )
    return resolved


def inspect_all_raw_bands() -> dict:
    """Print (and return) metadata for every raw band: driver, dtype,
    CRS, resolution, dimensions, nodata. Run this first so mismatches
    (wrong CRS, unexpected dtype, etc.) are visible before you spend
    time processing."""
    resolved = resolve_raw_band_paths()
    report = {}
    print(f"{'Band':<5} {'Driver':<14} {'Dtype':<8} {'CRS':<12} {'Res':<10} {'Size':<12} {'NoData'}")
    for band_key, path in resolved.items():
        info = rio.inspect_raster(path)
        report[band_key] = info
        crs_short = info["crs"].split(":")[-1] if ":" in info["crs"] else info["crs"]
        print(
            f"{band_key:<5} {info['driver']:<14} {info['dtype']:<8} "
            f"EPSG:{crs_short:<7} {info['resolution'][0]:<10.1f} "
            f"{info['width']}x{info['height']:<6} {info['nodata']}"
        )
    return report


# ---------------------------------------------------------------------------
# AOI: use a real one if present, otherwise derive one automatically
# ---------------------------------------------------------------------------
def get_or_build_aoi() -> gpd.GeoDataFrame:
    """Return an AOI GeoDataFrame.

    - If config.AOI_PATH exists, load and return it (this is the
      original V1 behaviour, unchanged).
    - Otherwise, build a rectangular AOI from the union of the training
      polygons' bounds (config.TRAINING_POLYGONS_PATH), expanded by
      config.AOI_BUFFER_METERS, save it to
      qgis/auto_generated_aoi.gpkg for your own inspection in QGIS, and
      return it. This does NOT invent any classification data -- it
      only defines a processing extent from polygons that already
      exist.
    """
    if config.AOI_PATH.exists():
        return gpd.read_file(config.AOI_PATH, layer=config.AOI_LAYER)

    if not config.TRAINING_POLYGONS_PATH.exists():
        raise FileNotFoundError(
            f"Neither an AOI file ({config.AOI_PATH}) nor training polygons "
            f"({config.TRAINING_POLYGONS_PATH}) were found. At least one is "
            f"required to define a processing extent."
        )

    print(
        f"No AOI file found at {config.AOI_PATH}. "
        f"Auto-deriving an AOI from the bounds of "
        f"{config.TRAINING_POLYGONS_PATH.name}, buffered by "
        f"{config.AOI_BUFFER_METERS} m."
    )

    polygons = gpd.read_file(config.TRAINING_POLYGONS_PATH, layer=config.TRAINING_POLYGONS_LAYER)
    minx, miny, maxx, maxy = polygons.total_bounds
    buffer = config.AOI_BUFFER_METERS
    from shapely.geometry import box

    aoi_geom = box(minx - buffer, miny - buffer, maxx + buffer, maxy + buffer)
    aoi = gpd.GeoDataFrame({"id": [1], "source": ["auto-generated"]}, geometry=[aoi_geom], crs=polygons.crs)

    out_path = config.QGIS_DIR / "auto_generated_aoi.gpkg"
    aoi.to_file(out_path, driver="GPKG")
    print(f"Saved auto-generated AOI -> {out_path} (open it in QGIS to check it looks right)")

    return aoi


# ---------------------------------------------------------------------------
# Step: Clip each raw band to the AOI, converting to float32 reflectance
# ---------------------------------------------------------------------------
def clip_band(band_key: str, raw_path=None, aoi=None) -> str:
    """Clip one raw Sentinel-2 band to the AOI. Converts to float32
    reflectance (if config.APPLY_REFLECTANCE_SCALING is True) and
    writes the result as a proper GeoTIFF via raster_io_v2.write_geotiff
    -- this is the exact point where the V1 pipeline silently inherited
    the JP2 driver; V2 does not.
    """
    raw_path = raw_path or rio.find_band_file(config.RAW_DIR, band_key, config.RAW_BANDS[band_key])
    out_path = config.CLIPPED_BANDS[band_key]

    aoi = aoi if aoi is not None else get_or_build_aoi()

    with rasterio.open(raw_path) as src:
        aoi_reproj = aoi.to_crs(src.crs)
        geoms = [geom.__geo_interface__ for geom in aoi_reproj.geometry]

        out_image, out_transform = rio_mask(src, geoms, crop=True, nodata=src.nodata)
        src_meta = src.meta.copy()
        src_nodata = src.nodata

    band_data = out_image[0]

    if config.APPLY_REFLECTANCE_SCALING:
        valid = True if src_nodata is None else (band_data != src_nodata)
        scaled = rio.apply_reflectance_scaling(
            band_data, config.REFLECTANCE_SCALE, config.RADIOMETRIC_OFFSET
        )
        # Preserve nodata as NaN so downstream steps can detect it
        # regardless of the original integer nodata value.
        scaled = np.where(valid, scaled, np.nan)
        out_dtype = "float32"
        out_nodata = np.nan
    else:
        scaled = band_data
        out_dtype = band_data.dtype
        out_nodata = src_nodata

    out_meta = src_meta.copy()
    out_meta["transform"] = out_transform

    rio.write_geotiff(out_path, scaled, out_meta, nodata=out_nodata, dtype=out_dtype)
    print(f"Clipped {band_key} -> {out_path} (dtype={out_dtype})")
    return str(out_path)


def clip_all_bands():
    # Compute the AOI once and reuse it for every band, instead of
    # re-deriving/reloading it per band.
    aoi = get_or_build_aoi()
    for band_key in config.BAND_ORDER:
        clip_band(band_key, aoi=aoi)


# ---------------------------------------------------------------------------
# Step: Resample 20 m bands to target resolution
# ---------------------------------------------------------------------------
def resample_band_to_target(band_key: str, target_resolution=None) -> str:
    target_resolution = target_resolution or config.TARGET_RESOLUTION
    in_path = config.CLIPPED_BANDS[band_key]
    resampling = Resampling[config.RESAMPLING_METHOD]

    with rasterio.open(in_path) as src:
        scale_x = src.res[0] / target_resolution
        scale_y = src.res[1] / target_resolution
        new_width = max(1, int(round(src.width * scale_x)))
        new_height = max(1, int(round(src.height * scale_y)))

        new_transform = src.transform * src.transform.scale(
            (src.width / new_width), (src.height / new_height)
        )

        data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=resampling,
        )
        meta = src.meta.copy()
        meta["transform"] = new_transform
        dtype = src.dtypes[0]
        nodata = src.nodata

    rio.write_geotiff(in_path, data, meta, nodata=nodata, dtype=dtype)
    print(f"Resampled {band_key} -> {target_resolution} m ({new_width}x{new_height})")
    return str(in_path)


def resample_all_required_bands():
    for band_key in config.BANDS_TO_RESAMPLE:
        resample_band_to_target(band_key)


# ---------------------------------------------------------------------------
# Step: Align every band to a common grid before stacking
# ---------------------------------------------------------------------------
def align_band_to_reference(band_key: str, reference_key: str):
    ref_path = config.CLIPPED_BANDS[reference_key]
    band_path = config.CLIPPED_BANDS[band_key]

    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(band_path) as src:
        if src.width == ref_width and src.height == ref_height and src.transform == ref_transform:
            return  # already aligned

        dtype = src.dtypes[0]
        nodata = src.nodata
        fill_value = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else (nodata or 0)
        destination = np.full((ref_height, ref_width), fill_value, dtype=dtype)

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=nodata,
            resampling=Resampling[config.RESAMPLING_METHOD],
        )
        meta = src.meta.copy()
        meta["transform"] = ref_transform

    rio.write_geotiff(band_path, destination, meta, nodata=nodata, dtype=dtype)
    print(f"Aligned {band_key} to {reference_key} grid")


def align_all_bands():
    reference_key = config.BAND_ORDER[0]
    for band_key in config.BAND_ORDER[1:]:
        align_band_to_reference(band_key, reference_key)


# ---------------------------------------------------------------------------
# Step: Build the 6-band stack
# ---------------------------------------------------------------------------
def build_band_stack() -> str:
    band_arrays = []
    ref_meta = None
    for band_key in config.BAND_ORDER:
        with rasterio.open(config.CLIPPED_BANDS[band_key]) as src:
            band_arrays.append(src.read(1))
            if ref_meta is None:
                ref_meta = src.meta.copy()

    stack = np.stack(band_arrays, axis=0).astype("float32")

    rio.write_geotiff(config.STACK_PATH, stack, ref_meta, nodata=np.nan, dtype="float32")
    with rasterio.open(config.STACK_PATH, "r+") as dst:
        dst.descriptions = tuple(config.BAND_ORDER)

    print(f"Built 6-band stack -> {config.STACK_PATH}")
    return str(config.STACK_PATH)


# ---------------------------------------------------------------------------
# Step: Spectral indices
# ---------------------------------------------------------------------------
def _safe_normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a - b) / denom
    return result  # NaNs (from NaN inputs or 0/0) propagate deliberately


def compute_indices():
    """Compute NDVI, NDWI, NDBI from the 6-band stack:

        NDVI = (B08 - B04) / (B08 + B04)
        NDWI = (B03 - B08) / (B03 + B08)
        NDBI = (B11 - B08) / (B11 + B08)

    Written as float32 GeoTIFFs via write_geotiff() -- this is exactly
    the step that crashed in V1.
    """
    with rasterio.open(config.STACK_PATH) as src:
        idx = {b: i + 1 for i, b in enumerate(config.BAND_ORDER)}
        b03 = src.read(idx["B03"])
        b04 = src.read(idx["B04"])
        b08 = src.read(idx["B08"])
        b11 = src.read(idx["B11"])
        meta = src.meta.copy()

    ndvi = _safe_normalized_difference(b08, b04)
    ndwi = _safe_normalized_difference(b03, b08)
    ndbi = _safe_normalized_difference(b11, b08)

    for arr, path, name in [
        (ndvi, config.NDVI_PATH, "NDVI"),
        (ndwi, config.NDWI_PATH, "NDWI"),
        (ndbi, config.NDBI_PATH, "NDBI"),
    ]:
        rio.write_geotiff(path, arr, meta, nodata=np.nan, dtype="float32")
        with rasterio.open(path, "r+") as dst:
            dst.set_band_description(1, name)
        print(f"Computed {name} -> {path}")

    return ndvi, ndwi, ndbi


# ---------------------------------------------------------------------------
# Step: Final 9-band feature stack
# ---------------------------------------------------------------------------
def build_final_stack() -> str:
    with rasterio.open(config.STACK_PATH) as src:
        band_stack = src.read().astype("float32")
        meta = src.meta.copy()

    with rasterio.open(config.NDVI_PATH) as src:
        ndvi = src.read(1).astype("float32")
    with rasterio.open(config.NDWI_PATH) as src:
        ndwi = src.read(1).astype("float32")
    with rasterio.open(config.NDBI_PATH) as src:
        ndbi = src.read(1).astype("float32")

    final_stack = np.concatenate(
        [band_stack, ndvi[np.newaxis, ...], ndwi[np.newaxis, ...], ndbi[np.newaxis, ...]],
        axis=0,
    )

    rio.write_geotiff(config.FINAL_STACK_PATH, final_stack, meta, nodata=np.nan, dtype="float32")
    with rasterio.open(config.FINAL_STACK_PATH, "r+") as dst:
        dst.descriptions = tuple(config.FEATURE_NAMES)

    print(f"Built final {final_stack.shape[0]}-band feature stack -> {config.FINAL_STACK_PATH}")
    return str(config.FINAL_STACK_PATH)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_full_preprocessing():
    print("Step 0: Validating raw band files...")
    inspect_all_raw_bands()

    print("\nStep 1: Clipping bands to AOI (and converting to reflectance)...")
    clip_all_bands()

    print("\nStep 2: Resampling 20 m bands to target resolution...")
    resample_all_required_bands()

    print("\nStep 3: Aligning all bands to a common grid...")
    align_all_bands()

    print("\nStep 4: Building 6-band stack...")
    build_band_stack()

    print("\nStep 5: Computing NDVI / NDWI / NDBI...")
    compute_indices()

    print("\nStep 6: Building final 9-band feature stack...")
    build_final_stack()

    print("\nPreprocessing complete.")

    final_info = rio.inspect_raster(config.FINAL_STACK_PATH)
    print(f"\nfinal_stack.tif: driver={final_info['driver']}, dtype={final_info['dtype']}, "
          f"bands={final_info['count']}, size={final_info['width']}x{final_info['height']}")

    return str(config.FINAL_STACK_PATH)
