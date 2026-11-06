"""
training_v2.py
===============
V2.0 changes from training.py:

  1. load_training_polygons() normalizes column names (strips
     whitespace, matches case-insensitively) so a field literally named
     "class_name " (as in your uploaded mumbai_lulc.gpkg) is found
     correctly instead of raising a false "missing column" error.

  2. The actual class_id -> class_name mapping is read from your
     polygon data at runtime and compared against config.CLASSES.
     Mismatches are reported clearly (not silently ignored, not fatal)
     so you always know whether the code is using what's really in
     your GeoPackage.

  3. extract_training_pixels() is NaN-aware: final_stack.tif's nodata
     pixels are now NaN (float32) rather than an arbitrary integer, and
     any pixel with a NaN in ANY of the 9 features is excluded before
     it ever reaches the CSV, rather than being cleaned out later.

Everything else (train/test split, RF/SVM/LogReg training, evaluation
metrics, model comparison, feature importance, joblib save/load) is
unchanged in behaviour from V1 -- that part of the pipeline was already
correct.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

import config_v2 as config


# ---------------------------------------------------------------------------
# Load + validate training polygons
# ---------------------------------------------------------------------------
def _normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Strip whitespace from column names (fixes 'class_name ' style
    QGIS export quirks) without touching the geometry column."""
    gdf = gdf.rename(columns={c: c.strip() for c in gdf.columns if isinstance(c, str)})
    return gdf


def load_training_polygons() -> gpd.GeoDataFrame:
    """Read the training polygon layer, tolerant of whitespace in
    column names. Raises a clear error only if the required columns
    are genuinely absent after normalization."""
    if not config.TRAINING_POLYGONS_PATH.exists():
        raise FileNotFoundError(
            f"Training polygons not found at {config.TRAINING_POLYGONS_PATH}. "
            "Set TRAINING_POLYGONS_PATH in config_v2.py to your polygon file."
        )
    gdf = gpd.read_file(config.TRAINING_POLYGONS_PATH, layer=config.TRAINING_POLYGONS_LAYER)
    gdf = _normalize_columns(gdf)

    missing = {config.CLASS_ID_FIELD, config.CLASS_NAME_FIELD} - set(gdf.columns)
    if missing:
        raise ValueError(
            f"Training polygon layer is missing expected column(s) even after "
            f"stripping whitespace: {missing}. Actual columns found: "
            f"{list(gdf.columns)}. Check CLASS_ID_FIELD / CLASS_NAME_FIELD in "
            f"config_v2.py."
        )
    return gdf


def report_actual_classes(polygons: gpd.GeoDataFrame) -> dict:
    """Print the class_id -> class_name mapping and polygon counts
    ACTUALLY present in the data, and flag any mismatch against
    config.CLASSES rather than silently trusting the config file."""
    actual = (
        polygons.groupby(config.CLASS_ID_FIELD)[config.CLASS_NAME_FIELD]
        .agg(lambda s: s.mode().iat[0])
        .to_dict()
    )
    counts = polygons.groupby(config.CLASS_ID_FIELD).size().to_dict()

    print("Classes found in training polygon data:")
    for cid in sorted(actual):
        cfg_name = config.CLASSES.get(cid)
        flag = "" if cfg_name and cfg_name.lower() == str(actual[cid]).lower() else "  <-- MISMATCH vs config.CLASSES"
        print(f"  class_id {cid}: '{actual[cid]}' ({counts[cid]} polygons)"
              f"{f' | config.py says: {cfg_name!r}' if flag else ''}{flag}")

    config_only = set(config.CLASSES) - set(actual)
    data_only = set(actual) - set(config.CLASSES)
    if config_only:
        print(f"  Note: config.CLASSES has class_id(s) {config_only} that do NOT "
              f"appear in your polygon data at all.")
    if data_only:
        print(f"  Note: your polygon data has class_id(s) {data_only} that are NOT "
              f"in config.CLASSES -- add them if you want proper labels on plots.")

    return actual


# ---------------------------------------------------------------------------
# Extract training pixels
# ---------------------------------------------------------------------------
def extract_training_pixels() -> pd.DataFrame:
    """For every training polygon, extract the 9 feature values for
    every pixel inside it and label it with class_id. Skips pixels that
    are NaN in any feature (nodata / outside valid data), since
    final_stack.tif's nodata pixels are NaN in V2."""
    polygons = load_training_polygons()
    report_actual_classes(polygons)

    with rasterio.open(config.FINAL_STACK_PATH) as src:
        polygons = polygons.to_crs(src.crs)
        all_rows = []

        for class_id, class_group in polygons.groupby(config.CLASS_ID_FIELD):
            geoms = [geom.__geo_interface__ for geom in class_group.geometry]

            class_mask = geometry_mask(
                geoms,
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
            )

            rows_idx, cols_idx = np.where(class_mask)
            n_candidate = len(rows_idx)
            if n_candidate == 0:
                print(f"Warning: class_id {class_id} produced 0 pixels -- polygon may fall "
                      f"outside the raster extent (check CRS/AOI).")
                continue

            feature_values = np.zeros((n_candidate, len(config.FEATURE_NAMES)), dtype="float32")
            for band_i in range(len(config.FEATURE_NAMES)):
                band_data = src.read(band_i + 1)
                feature_values[:, band_i] = band_data[rows_idx, cols_idx]

            valid = ~np.isnan(feature_values).any(axis=1)
            n_valid = valid.sum()

            class_df = pd.DataFrame(feature_values[valid], columns=config.FEATURE_NAMES)
            class_df["class"] = class_id
            all_rows.append(class_df)

            class_name = config.CLASSES.get(class_id, "?")
            print(f"class_id {class_id} ({class_name}): {n_valid} valid pixels "
                  f"({n_candidate - n_valid} excluded as nodata/outside raster)")

    if not all_rows:
        raise ValueError(
            "No training pixels were extracted for ANY class. This usually means "
            "the training polygons don't overlap final_stack.tif's extent -- check "
            "that both are in a compatible area and CRS."
        )

    training_df = pd.concat(all_rows, ignore_index=True)
    return training_df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any remaining inf values, exact duplicates, and classes
    with too few samples for a stratified split."""
    before = len(df)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()

    counts = df["class"].value_counts()
    valid_classes = counts[counts >= 2].index
    dropped_classes = set(counts.index) - set(valid_classes)
    if dropped_classes:
        print(f"Dropping class(es) with <2 samples (cannot stratify-split): {dropped_classes}")
    df = df[df["class"].isin(valid_classes)]

    after = len(df)
    print(f"Cleaned training data: {before} -> {after} rows ({before - after} removed)")
    return df.reset_index(drop=True)


def subsample_training_data(df: pd.DataFrame, max_per_class=None, random_state=None) -> pd.DataFrame:
    """Cap the number of pixels per class before training.

    WHY THIS MATTERS (not just a performance shortcut): pixels inside
    the same training polygon are highly spatially autocorrelated --
    a 200x200 m built-up polygon at 10 m resolution contributes 400
    nearly-identical pixels. Using all of them (i) massively
    overweights whichever class happens to have the largest polygons,
    and (ii) makes RBF-kernel SVM training scale very badly: sklearn's
    SVC is roughly O(n^2)-O(n^3), so 70,000+ raw pixels (as produced by
    your actual training.gpkg) can take a very long time to fit.
    Capping per class both fixes the runtime problem and gives a more
    balanced, less redundant training set.

    Set config.MAX_SAMPLES_PER_CLASS = None to disable this and use
    every extracted pixel (only recommended for small training sets).
    """
    max_per_class = max_per_class if max_per_class is not None else config.MAX_SAMPLES_PER_CLASS
    random_state = random_state if random_state is not None else config.RANDOM_STATE

    if max_per_class is None:
        return df

    before = len(df)
    rng = np.random.default_rng(random_state)
    sampled_frames = []
    for class_value, group in df.groupby("class"):
        n = min(len(group), max_per_class)
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        sampled_frames.append(df.loc[idx])
    df = pd.concat(sampled_frames, ignore_index=True)
    after = len(df)
    if after < before:
        print(f"Subsampled training data to at most {max_per_class} pixels/class: "
              f"{before} -> {after} rows")
    return df


def save_training_csv(df: pd.DataFrame, path=None) -> str:
    path = path or config.TRAINING_CSV_PATH
    df.to_csv(path, index=False)
    print(f"Saved training data -> {path} ({len(df)} rows)")
    return str(path)


def load_training_csv(path=None) -> pd.DataFrame:
    path = path or config.TRAINING_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found at {path}. Run notebook 02 first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def make_train_test_split(df: pd.DataFrame, feature_names=None):
    feature_names = feature_names or config.FEATURE_NAMES
    X = df[feature_names]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Train models
# ---------------------------------------------------------------------------
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    return rf


def train_svm(X_train, y_train) -> SVC:
    svm = SVC(**config.SVM_PARAMS)
    svm.fit(X_train, y_train)
    return svm


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    lr = LogisticRegression(**config.LR_PARAMS)
    lr.fit(X_train, y_train)
    return lr


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="model", plot_confusion=False):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1 (weighted):        {f1:.4f}")
    print(f"Cohen's Kappa:        {kappa:.4f}")
    print("\n" + report)

    if plot_confusion:
        plot_confusion_matrix(cm, model_name=model_name, class_ids=sorted(y_test.unique()))

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_confusion_matrix(cm, model_name="model", class_ids=None, save_path=None):
    class_ids = class_ids or sorted(config.CLASSES.keys())
    class_labels = [config.CLASSES.get(c, str(c)) for c in class_ids]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels[: cm.shape[0]],
        yticklabels=class_labels[: cm.shape[0]],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    save_path = save_path or config.CONFUSION_MATRIX_PNG
    plt.savefig(save_path, dpi=150)
    print(f"Saved confusion matrix plot -> {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Compare models
# ---------------------------------------------------------------------------
def compare_models(results: list) -> pd.DataFrame:
    rows = [
        {
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "Kappa": r["kappa"],
        }
        for r in results
    ]
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


def save_model_comparison(comparison_df: pd.DataFrame, path=None) -> str:
    path = path or config.MODEL_COMPARISON_CSV
    comparison_df.to_csv(path, index=False)
    print(f"Saved model comparison -> {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def get_feature_importance(rf_model: RandomForestClassifier, feature_names=None) -> pd.DataFrame:
    feature_names = feature_names or config.FEATURE_NAMES
    return (
        pd.DataFrame({"feature": feature_names, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(importance_df: pd.DataFrame, save_path=None):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", color="steelblue")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    save_path = save_path or config.FEATURE_IMPORTANCE_PNG
    plt.savefig(save_path, dpi=150)
    print(f"Saved feature importance plot -> {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Save / load models
# ---------------------------------------------------------------------------
def save_model(model, path):
    joblib.dump(model, path)
    print(f"Saved model -> {path}")


def load_model(path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}.")
    return joblib.load(path)
