# COMP4423 Assignment 2 — Campus Vegetation Recognition

This repository contains a **traditional (non-deep-learning)** computer-vision pipeline for campus vegetation classification using handcrafted features (HSV histogram + LBP + HOG), classical ML models (SVM / Random Forest / Gradient Boosting), and a Streamlit demo app.

## README review (quick notes)

- ✅ The project structure is clear and implementation files are well separated by responsibility.
- ✅ The workflow order (build split → train → evaluate → app) is correct and reproducible.
- ⚠️ In this environment, OpenCV may fail with `ImportError: libGL.so.1` unless system GUI libs are installed.
- ℹ️ A full assignment report draft is provided in `Assignment2_Report_Markdown.md`.

## Project structure

```text
.
├── app/
│   └── app.py                    # Streamlit UI for single-image prediction
├── img-train/                    # Input dataset (one folder per class)
├── src/
│   ├── build_dataset.py          # Dataset validation + stratified split creation
│   ├── features.py               # Preprocessing + feature extraction
│   ├── train.py                  # Model selection and artifact export
│   ├── evaluate.py               # Test split evaluation + reports
│   ├── predict.py                # CLI single-image inference
│   └── config.py                 # Shared project constants
├── outputs/
├── models/
├── requirements.txt
└── Assignment2_Report_Markdown.md
```

## Requirements

- Python 3.10+
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset format

Training images are expected under `img-train/` in class-named subfolders:

```text
img-train/
├── class-a/
│   ├── img1.png
│   └── ...
├── class-b/
│   ├── img1.png
│   └── ...
└── ...
```

## End-to-end workflow

Run commands from project root.

### 1) Build validated train/val/test split

```bash
python src/build_dataset.py
```

What it does:
- validates images (decode check),
- detects duplicate-content files by SHA-1,
- creates stratified split (about 70/15/15),
- writes:
  - `data/splits/split.csv`
  - `data/splits/class_counts.csv`

### 2) Train and select best model

```bash
python src/train.py
```

Writes:
- `models/best_model.pkl`
- `models/label_encoder.pkl`
- `outputs/training_summary.json`

### 3) Evaluate on test split

```bash
python src/evaluate.py
```

Optional misclassification copy:

```bash
python src/evaluate.py --copy-errors 20
```

Writes:
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/error_cases.csv`
- `outputs/error_cases/` (when enabled)

### 4) Predict a single image from CLI

```bash
python src/predict.py --image path/to/image.png
```

Optional arguments:
- `--model models/best_model.pkl`
- `--label-encoder models/label_encoder.pkl`
- `--top-k 5`

## Run the Streamlit app

```bash
streamlit run app/app.py
```

In the app:
1. (Optional) adjust model and encoder paths.
2. Upload one image (`jpg/jpeg/png/bmp/webp`).
3. Click **Predict** to view class prediction and probabilities.

## Reproducibility notes

- Random seed is fixed to `42` in split/model routines.
- Split ratios are approximately `70/15/15`.
- Model/split artifact paths are centralized in code.

## Troubleshooting

- **`Split file not found`**: run `python src/build_dataset.py` first.
- **`Model file not found`**: run `python src/train.py` first.
- **`ImportError: libGL.so.1`** (Linux headless env): install OpenCV runtime dependencies or use a compatible environment.
- **Decode issues**: verify images are valid and not corrupted.

## Quick start

```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/train.py
python src/evaluate.py
streamlit run app/app.py
```
