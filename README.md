# comp4423-computer-vision-asgmt-2

A classical computer-vision pipeline for **leaf image classification** using handcrafted features (HSV histogram + LBP + HOG) and traditional ML models (SVM, Random Forest, Gradient Boosting), plus a Streamlit demo app.

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
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+ (recommended)
- pip

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

The repository already includes a dataset in this format.

## End-to-end workflow

Run the following commands from the project root.

### 1) Build validated train/val/test split

```bash
python src/build_dataset.py
```

What it does:
- Verifies files are decodable images.
- Computes SHA-1 hashes and reports duplicate-content files.
- Builds a stratified split (approx. 70% train / 15% val / 15% test).
- Writes:
  - `data/splits/split.csv`
  - `data/splits/class_counts.csv`

### 2) Train and select the best model

```bash
python src/train.py
```

What it does:
- Extracts fused handcrafted features per image.
- Trains multiple model families + hyperparameter configs.
- Selects best candidate on validation accuracy (macro-F1 tie-break).
- Writes:
  - `models/best_model.pkl`
  - `models/label_encoder.pkl`
  - `outputs/training_summary.json`

### 3) Evaluate on test split

```bash
python src/evaluate.py
```

Optional: copy up to N misclassified images into an output folder.

```bash
python src/evaluate.py --copy-errors 20
```

Evaluation outputs:
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/error_cases.csv`
- `outputs/error_cases/` (when `--copy-errors > 0`)

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
1. (Optional) adjust model and label-encoder paths.
2. Upload an image (`jpg/jpeg/png/bmp/webp`).
3. Click **Predict** to get the predicted class and top probabilities (if supported by the model).

## Reproducibility notes

- Default random seed is fixed to `42` for split/model routines.
- Current split ratios in code are 70/15/15.
- Model and split paths are centralized in `src/config.py`.

## Troubleshooting

- **`Split file not found`**: run `python src/build_dataset.py` first.
- **`Model file not found`**: run `python src/train.py` first.
- **Prediction/evaluation decode issues**: verify image files are valid and not corrupted.

## Quick start (minimal)

```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/train.py
python src/evaluate.py
streamlit run app/app.py
```
