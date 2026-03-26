# COMP4423 – Computer Vision (2025/26 Semester 2)
## Assignment 2 Report (Markdown Version)
### Campus Vegetation Recognition

> **Student Name:** `<YOUR_NAME>`  
> **Student ID:** `<YOUR_STUDENT_ID>`  
> **Submission Date:** `March 2026`  
> **Deadline:** **11:59 PM, Tuesday, 24 March 2026**

---

## 1. Task Overview
This project implements a complete **traditional machine learning** pipeline for campus vegetation recognition using a self-collected dataset. The assignment requirement to avoid deep learning was strictly followed. The pipeline includes:

1. Class definition and collection planning.
2. Dataset building, validation, and labeling.
3. Feature extraction with handcrafted descriptors.
4. Model training and model selection among classical classifiers.
5. Held-out test evaluation and error analysis.
6. A simple local application for image-based prediction.

---

## 2. Task 1 — Class Definition & Data Collection Plan

### 2.1 Defined vegetation classes (8 classes)
The dataset contains the following classes:

1. `areca-palm`
2. `buddhist-pine`
3. `chinese-hibiscus`
4. `grass`
5. `palm`
6. `podocarpus`
7. `podocarpus-macrophyllus`
8. `tropical-almond-tree`

### 2.2 Data collection plan
During on-campus collection, each class should be photographed with diversity in:

- **Location:** at least 2–3 different campus spots per class when possible.
- **Viewpoint:** front/side/oblique/top-down where feasible.
- **Distance:** close-up leaf/flower texture + mid-range + whole plant context.
- **Lighting/time:** morning, noon, late afternoon; cloudy/sunny conditions.
- **Scene complexity:** simple and cluttered backgrounds.
- **Occlusion:** partial occlusion by branches, fences, people, or other plants.

### 2.3 Ground-truth practice
Nearby **plant name plates** should be used as class verification. A practical workflow is:

1. Capture plate image for traceability.
2. Capture multiple plant photos immediately after verification.
3. Repeat from multiple positions and distances.

---

## 3. Task 2 — Dataset Building & Labeling

### 3.1 Dataset structure
The project uses class-folder organization:

```text
img-train/
├── areca-palm/
├── buddhist-pine/
├── chinese-hibiscus/
├── grass/
├── palm/
├── podocarpus/
├── podocarpus-macrophyllus/
└── tropical-almond-tree/
```

A split builder script validates images, checks duplicates (SHA-1), and creates stratified train/val/test CSV split files.

### 3.2 Image count summary

| Class | Full | Train | Val | Test |
|---|---:|---:|---:|---:|
| areca-palm | 13 | 9 | 2 | 2 |
| buddhist-pine | 7 | 5 | 1 | 1 |
| chinese-hibiscus | 6 | 4 | 1 | 1 |
| grass | 6 | 4 | 1 | 1 |
| palm | 8 | 6 | 1 | 1 |
| podocarpus | 8 | 6 | 1 | 1 |
| podocarpus-macrophyllus | 10 | 7 | 1 | 2 |
| tropical-almond-tree | 8 | 5 | 2 | 1 |
| **Total** | **66** | **46** | **10** | **10** |

### 3.3 Sanity checks
Implemented checks include:

- corrupted/unreadable image detection (OpenCV decode test),
- SHA-1 duplicate-content detection,
- split integrity checks (`train`, `val`, `test` only),
- required CSV columns validation (`filepath`, `label`, `split`).

---

## 4. Task 3 — Traditional ML Classifier Training

### 4.1 Why these features?
A **fused handcrafted feature** vector is used:

- **HSV color histogram (16×16×16):** captures color distribution robustly in HSV space.
- **LBP histogram:** captures local texture patterns helpful for leaves/grass surfaces.
- **HOG descriptor:** captures shape and edge structure (leaf boundaries, branch patterns).

This combination improves robustness across appearance variations compared with a single feature family.

### 4.2 Feature preprocessing
- Input images are resized to a fixed shape before extraction.
- Grayscale conversion is used for LBP and HOG branches.
- Feature vectors are concatenated into one vector (`n_features = 12,206`).

### 4.3 Candidate models (no deep learning)
Three classical model families were evaluated:

- **SVM** (linear and RBF kernel variants),
- **Random Forest**,
- **Gradient Boosting**.

Model selection criterion:
1. highest validation accuracy,
2. macro-F1 tie-break when accuracy ties.

### 4.4 Training results (validation)
Best model:

- **Model:** Random Forest
- **Hyperparameters:** `n_estimators=200`, `max_depth=None`, `min_samples_split=2`
- **Validation Accuracy:** `0.90`
- **Validation Macro-F1:** `0.85`

Artifacts generated:

- `models/best_model.pkl`
- `models/label_encoder.pkl`
- `outputs/training_summary.json`

---

## 5. Task 4 — Evaluation & Error Analysis

### 5.1 Test-set quantitative results
From held-out test set (`n=10`):

- **Accuracy:** `1.00`
- **Macro-F1:** `1.00` (all per-class F1 values are 1.0)
- **Confusion matrix:** identity matrix (all test samples correctly classified)

Generated files:

- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/error_cases.csv`

### 5.2 Qualitative outcomes
- Current run has **0 misclassified test images**, so `error_cases.csv` is empty.
- This indicates good separability for the current split, but the test size is small.

### 5.3 Failure mode discussion (feature limitations)
Even though this run is perfect on the held-out split, expected failure modes in real deployment include:

1. **Background bias**  
   Color histogram may capture background greenery/soil rather than plant-only cues.

2. **Lighting shifts**  
   HSV histograms can still drift under severe illumination changes and shadows.

3. **Similar-looking species**  
   Species with near-identical color and local texture may confuse LBP + HSV.

4. **Viewpoint and scale changes**  
   HOG is not fully invariant to large viewpoint/scale changes, especially with fixed resizing.

5. **Blur and motion noise**  
   LBP and HOG quality degrades when fine texture/edges are blurred.

### 5.4 Recommended robustness improvements
- Increase data scale per class and scene diversity.
- Add segmentation or background suppression before feature extraction.
- Add color constancy preprocessing.
- Use k-fold cross-validation for more stable performance estimates.

---

## 6. Task 5 — Application
A local Streamlit app (`app/app.py`) is provided.

### 6.1 Functionality
- Upload an image (`jpg`, `jpeg`, `png`, `bmp`, `webp`)
- Predict plant class using saved model and label encoder
- Display top-k probabilities (if model supports probability outputs)

### 6.2 How to run
```bash
pip install -r requirements.txt
python src/build_dataset.py
python src/train.py
python src/evaluate.py
streamlit run app/app.py
```

---

## 7. Reproducibility
- Fixed random seed (`42`) for splitting and model routines.
- Split strategy: stratified approx. `70/15/15`.
- Consistent preprocessing/feature extraction functions used across training, evaluation, and inference.
- Saved artifacts and JSON summaries support reproducible reporting.

---

## 8. Generative AI Collaboration Reflection
Generative AI was used as a coding assistant for:
- suggesting training/evaluation script structure,
- checking parameter grids and logging formats,
- improving report organization.

### 8.1 Observed limitations of AI suggestions
- Some generated suggestions were too generic and did not match assignment constraints.
- Some draft code omitted practical safeguards (file checks, split-column validation).
- Some suggestions overestimated expected performance reliability on small test sets.

### 8.2 Corrections made through human reasoning/testing
- Explicitly enforced **non-deep-learning** pipeline.
- Added/kept robust input validation and error handling.
- Interpreted perfect test accuracy cautiously due to limited sample size.

---

## 9. Conclusion
This assignment demonstrates a full classical CV + ML workflow for campus vegetation recognition, from collection planning to deployable local inference. The chosen handcrafted feature fusion with Random Forest achieved strong validation performance and perfect current split test results. Future work should focus on larger datasets and harder distribution shifts for stronger generalization evidence.

---

## 10. Submission Checklist
- [ ] Convert this markdown to PDF and rename as:  
  `Assignment2_Report_<your_ID>_<your_name>.pdf`
- [ ] Zip code + dataset and rename as:  
  `Assignment2_Code_<your_ID>_<your_name>.zip`
- [ ] Upload both files to Blackboard before deadline.

