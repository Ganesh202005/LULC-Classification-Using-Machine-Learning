# 🌍 LULC Classification Using Machine Learning

**Land Use and Land Cover Classification using Sentinel-2 Satellite Imagery and Machine Learning**

This project uses satellite imagery and machine learning to classify a study area into five land-cover categories: **Water, Built-up, Vegetation, Agriculture and Barren Land**.

It combines remote sensing, spectral indices, training polygons and machine learning algorithms to produce a classified Land Use/Land Cover (LULC) map.

The project compares three machine learning models:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

Random Forest achieved the best overall accuracy in the model comparison.

---

## 📌 Project Overview

Satellite images contain information about the Earth's surface in different spectral bands. Different land-cover types reflect electromagnetic energy differently.

For example:
- Water generally absorbs much of the near-infrared energy.
- Vegetation reflects strongly in the near-infrared band.
- Built-up areas have different spectral characteristics from vegetation and water.

This project uses these spectral differences to train machine learning models to recognise land-cover categories.

**In simple terms:** We provide the model with examples of known land-cover types, let it learn their spectral patterns, and then use what it has learned to classify the remaining satellite image.

## ⚙️ How the Workflow Works

The project is organised into Python scripts and Jupyter notebooks. Each part has a specific responsibility.

### 1. Satellite Data Preparation

The workflow begins with Sentinel-2 Level-2A satellite imagery.

Six spectral bands are used:

| Band | Description | Main use |
|---|---|---|
| B02 | Blue | Water and surface features |
| B03 | Green | Vegetation and water features |
| B04 | Red | Vegetation and land features |
| B08 | Near Infrared | Vegetation analysis |
| B11 | Short-wave Infrared 1 | Built-up and moisture-related features |
| B12 | Short-wave Infrared 2 | Land and surface characteristics |

The preprocessing script prepares the imagery by:
- Checking the required input bands.
- Clipping the imagery to the study area.
- Aligning the bands to a common grid.
- Resampling the 20 m bands to the 10 m working resolution.
- Creating a multi-band satellite stack.

This ensures that all bands can be analysed together, pixel by pixel.

### 2. Creating Spectral Indices

Along with the six original bands, the workflow calculates three spectral indices:

| Index | What it helps identify |
|---|---|
| NDVI | Vegetation characteristics |
| NDWI | Water-related characteristics |
| NDBI | Built-up characteristics |

These indices are calculated using combinations of satellite bands.

The final feature stack contains **nine features: six spectral bands and three indices**.

### 3. How Does the Model Learn from Training Polygons?

This is one of the most important parts of the project.

Before a machine learning model can classify an image, it needs examples of what the different land-cover classes look like.

These examples are created using **training polygons** in QGIS.

Imagine opening the satellite image in QGIS and drawing polygons over locations that you know represent:

- Water
- Built-up areas
- Vegetation
- Agriculture
- Barren land

Each polygon is assigned a class ID and class name.

For example:

| Polygon | Class ID | Class name |
|---|---:|---|
| Polygon 1 | 1 | Water |
| Polygon 2 | 2 | Built-up |
| Polygon 3 | 3 | Vegetation |
| Polygon 4 | 4 | Agriculture |
| Polygon 5 | 5 | Barren |

**What happens next?**

1. The training polygons are loaded from the GeoPackage.
2. The program identifies the satellite pixels falling inside each polygon.
3. For every selected pixel, it extracts the values of all nine features.
4. Each pixel is paired with the class label of its polygon.
5. These labelled pixel samples become the training dataset.

For example, a pixel inside a Water polygon might have a particular combination of Blue, Green, Red, NIR, SWIR and spectral-index values. A pixel inside a Vegetation polygon will generally have a different combination.

The model learns these relationships:

**Satellite pixel values + known class labels → Machine Learning model**

After learning from the labelled samples, the model can predict the class of pixels outside the training polygons.

In this way, the polygons act like a teacher: they provide examples, while the machine learning algorithm learns the patterns.

### 4. Model Training and Comparison

The extracted pixel samples are used to train three machine learning algorithms:

| Model | Basic idea |
|---|---|
| Random Forest | Combines multiple decision trees to make a prediction. |
| SVM | Finds boundaries that separate different classes. |
| Logistic Regression | Estimates the probability that a sample belongs to a particular class. |

The dataset is divided into training and testing samples using a stratified 70:30 split.

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Cohen's Kappa

The model with the strongest overall performance is selected for the final raster classification.

### 5. Final LULC Classification

The trained Random Forest model is loaded and applied to the prepared satellite image.

It examines the nine feature values for each valid pixel and predicts one of the five land-cover classes.

The predictions are processed in chunks to handle the large raster efficiently.

The final result is a classified LULC raster that can be opened and visualised in QGIS.

---

## 🗂️ Project Structure

```text
LULC-Classification-Using-Machine-Learning/
│
├── models/
│   ├── random_forest.joblib
│   ├── svm.joblib
│   └── logistic_regression.joblib
│
├── notebooks/
│   ├── 01_data_preparation_v2.ipynb
│   ├── 02_extract_training_data_v2.ipynb
│   ├── 03_train_models_v2.ipynb
│   └── 04_classify_raster_v2.ipynb
│
├── outputs/
│   ├── classified_map.tif
│   ├── confusion_matrix.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── feature_set_experiment.csv
│   └── model_comparison.csv
│
├── qgis/
│   ├── training.gpkg
│   └── auto_generated_aoi.gpkg
│
├── scripts/
│   ├── config_v2.py
│   ├── preprocessing_v2.py
│   ├── raster_io_v2.py
│   ├── training_v2.py
│   ├── classification_v2.py
│   └── README_V2.md
│
├── .gitignore
└── LICENSE
```

**Note:** The raw satellite imagery, processed raster files and extracted training-pixel CSV are not included in this GitHub repository because of their large file sizes. They must be prepared locally before running the complete workflow.

---

## 📥 Setting Up the Missing Data

To run the complete project, create the following folders locally:

```text
data/
├── raw/
│   └── sentinel2/
│
├── processed/
│
└── training/
```

### 1. Download Sentinel-2 Imagery

The satellite data used in this project is Sentinel-2 Level-2A imagery.

You can obtain Sentinel-2 data through the **Copernicus Data Space Ecosystem**.

Basic download procedure:

1. Open the Copernicus Browser.
2. Register for a free account and sign in.
3. Search for the Mumbai study area.
4. Select the Sentinel-2 data collection and Level-2A product.
5. Choose the acquisition date and satellite tile matching the imagery expected by the project configuration.
6. Download the required spectral bands: B02, B03, B04, B08, B11 and B12.
7. Place the downloaded band files inside `data/raw/sentinel2/`.

**Important:** The current configuration expects a particular Sentinel-2 scene and band filenames. Downloading a different date or tile may require updating the filenames and scene settings in `config_v2.py`.

### 2. Training Polygons

The project uses `qgis/training.gpkg` as its training polygon dataset.

If you want to recreate or modify the training data:

1. Open the satellite imagery in QGIS.
2. Create a polygon layer.
3. Digitise representative areas for all five land-cover classes.
4. Add the `class_id` and `class_name` attributes.
5. Assign the correct class ID and name to each polygon.
6. Save the layer as a GeoPackage named `training.gpkg`, using the layer name `training_polygons`.

The five class IDs must follow the configuration:

| Class ID | Class |
|---:|---|
| 1 | Water |
| 2 | Built-up |
| 3 | Vegetation |
| 4 | Agriculture |
| 5 | Barren |

The workflow extracts the satellite pixel values from these polygons automatically.

### 3. Processed Data

You do not need to manually create every processed raster.

The preprocessing workflow generates the clipped and aligned bands, spectral indices and final feature stack from the raw satellite imagery.

If a separate study-area boundary is not supplied, the current preprocessing code can generate a rectangular AOI from the combined bounds of the training polygons, with a buffer.

---

## 📊 Project Results

### 1. Model Comparison

The three models were evaluated using the same validation approach.

| Model | Accuracy | Precision | Recall | F1-score | Kappa |
|---|---:|---:|---:|---:|---:|
| Random Forest | **96.87%** | 96.87% | 96.87% | 96.87% | 0.9608 |
| SVM | 95.18% | 95.26% | 95.18% | 95.19% | 0.9397 |
| Logistic Regression | 87.82% | 87.90% | 87.82% | 87.79% | 0.8478 |

Random Forest achieved the highest accuracy and Cohen's Kappa among the three tested models.

### 2. Final Classified LULC Map

The final classified raster represents the predicted land-cover distribution across the study area.

The five classes are:

- 🔵 Water
- 🔴 Built-up
- 🟢 Vegetation
- 🟡 Agriculture
- 🟤 Barren

Open `outputs/classified_map.tif` in QGIS to explore the classification spatially.

### 3. Confusion Matrix

The confusion matrix compares the actual validation labels with the model's predicted labels.

- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal values represent correctly classified samples.
- Values outside the diagonal represent classification errors.

The model correctly classified 4,359 out of 4,500 validation samples, giving an accuracy of approximately **96.87%**.

The matrix also helps identify confusion between classes. For example, some Built-up samples were predicted as Agriculture, and some Agriculture samples were predicted as Built-up.

### 4. Feature Importance

Random Forest provides a feature-importance score that indicates how much each feature contributed to the model's tree-based decisions.

The most important features in this experiment were:

| Feature | Importance |
|---|---:|
| B02 | 0.1889 |
| B04 | 0.1527 |
| NDVI | 0.1525 |
| NDWI | 0.1234 |
| B11 | 0.1171 |
| B03 | 0.0808 |
| B12 | 0.0800 |
| NDBI | 0.0569 |
| B08 | 0.0477 |

B02 (Blue), B04 (Red) and NDVI had the highest importance scores in this experiment.

These scores describe the model's relative feature importance; they do not establish that a feature independently causes a particular classification.

### 5. Feature Set Experiment

An additional experiment compared Random Forest performance using different feature combinations.

| Feature set | Accuracy |
|---|---:|
| Spectral bands only | 96.98% |
| All features (bands + indices) | 96.87% |
| Spectral indices only | 85.69% |

The spectral-bands-only experiment performed slightly better in this particular validation split. However, the final classification workflow uses all nine features.

---

## 📁 Understanding the Output Files

| File | Purpose |
|---|---|
| `classified_map.tif` | Final georeferenced LULC classification raster. |
| `confusion_matrix.png` | Visual comparison of actual and predicted validation classes. |
| `feature_importance.csv` | Numerical feature-importance values. |
| `feature_importance.png` | Visual ranking of the Random Forest features. |
| `model_comparison.csv` | Performance comparison of Random Forest, SVM and Logistic Regression. |
| `feature_set_experiment.csv` | Comparison of bands-only, indices-only and combined features. |
| `random_forest.joblib` | Saved Random Forest model. |
| `svm.joblib` | Saved SVM model. |
| `logistic_regression.joblib` | Saved Logistic Regression model. |

The GeoPackages in the `qgis/` folder store the training polygons and the automatically generated AOI.

---

## 🧪 Important Validation Note

The reported validation scores come from a stratified random pixel-level train/test split.

Because nearby pixels can have similar spectral characteristics, a random pixel split may produce more optimistic results than testing on completely separate geographic areas.

A future improvement would be to use spatially separated training and validation regions to assess how well the model generalises to new locations.

---

## 🎯 Conclusion

This project demonstrates how satellite remote sensing and machine learning can be combined to perform LULC classification.

It shows the complete workflow, from Sentinel-2 image preparation and QGIS training polygon creation to feature extraction, model training, accuracy assessment and final raster classification.

Among the three tested algorithms, Random Forest achieved the best overall performance, with an accuracy of **96.87%**.

The project also demonstrates the importance of training data, spectral features and validation metrics in developing a machine-learning-based geospatial classification workflow.

**The main objective is not just to produce a classified map, but to demonstrate how machine learning can learn from labelled geographic samples and apply that knowledge to satellite imagery.**

---

## 🚀 Future Improvements

Some possible extensions include:

- Spatial train/test validation.
- Testing additional machine learning algorithms.
- Improving training polygon coverage and class balance.
- Comparing results across different dates.
- Exploring deep-learning-based land-cover classification.
- Integrating the classification workflow into a QGIS plugin.

---

**Built with:** Python · QGIS · Sentinel-2 · Rasterio · GeoPandas · Scikit-learn · Jupyter Notebook
