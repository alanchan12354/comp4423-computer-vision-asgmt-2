# COMP4423 – Computer Vision (2025/26 Semester 2)
## Assignment 2 Report (Markdown)
### Campus Vegetation Recognition

> **Deadline reference:** 11:59 PM, Tuesday, 24 March 2026  
> **Author:** `<your_name>`  
> **Student ID:** `<your_student_id>`

---

## 1) Task Overview
This project implements a complete **traditional machine learning** pipeline for campus vegetation recognition using a self-collected dataset and handcrafted image features. The assignment explicitly disallows deep learning; therefore, the system is based on **HSV color histograms + LBP + HOG** feature extraction with classical classifiers (**SVM, Random Forest, Gradient Boosting**) and includes a local prediction application built with Streamlit.

The implementation in this repository follows an end-to-end workflow:
1. Validate and split dataset (`src/build_dataset.py`)
2. Train/select model (`src/train.py`)
3. Evaluate on held-out test set (`src/evaluate.py`)
4. Provide CLI and UI inference (`src/predict.py`, `app/app.py`)

---

## 2) Task 1 — Class Definition & Data Collection Plan

### 2.1 Selected classes (8 classes)
The project uses at least 8 plant categories:
1. areca-palm  
2. buddhist-pine  
3. chinese-hibiscus  
4. grass  
5. palm  
6. podocarpus  
7. podocarpus-macrophyllus  
8. tropical-almond-tree

### 2.2 Data collection strategy
For each class, images should be collected with the following variation dimensions:
- **Location diversity:** same species from different campus spots
- **Viewpoint diversity:** front/side/top-ish leaf clusters, near/far shots
- **Lighting diversity:** morning/noon/late afternoon, sunny/cloudy/shadow
- **Background diversity:** building walls, roads, lawns, pedestrians, mixed clutter
- **Scale/occlusion diversity:** partial leaves, overlapping branches, dense/sparse foliage

### 2.3 Ground-truth protocol
When available, nearby **plant name plates** are used to confirm the species label before taking multiple plant photos. This reduces label ambiguity for visually similar species.

---

## 3) Task 2 — Dataset Building & Labeling

### 3.1 Dataset structure
The dataset is organized in class-per-folder format:

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

### 3.2 Image counts
Current class counts in `img-train/`:

| Class | # Images |
|---|---:|
| areca-palm | 13 |
| buddhist-pine | 7 |
| chinese-hibiscus | 6 |
| grass | 6 |
| palm | 8 |
| podocarpus | 8 |
| podocarpus-macrophyllus | 10 |
| tropical-almond-tree | 8 |
| **Total** | **66** |

### 3.3 Labeling and sanity checks
The dataset builder script (`src/build_dataset.py`) includes these checks:
- **Corruption/unreadable detection:** verifies OpenCV can decode each file
- **Duplicate-content scan:** computes SHA-1 hash and reports duplicate groups
- **Stratified splitting:** creates train/val/test split (approx. 70/15/15)
- **Count report:** outputs per-class distribution summary

Expected generated files after running builder:
- `data/splits/split.csv`
- `data/splits/class_counts.csv`

---

## 4) Task 3 — Traditional ML Classifier Training

### 4.1 Why these features?
The model uses feature fusion from `src/features.py`:
- **HSV histogram:** captures color distribution (useful for green hue ranges, flower color cues)
- **LBP (Local Binary Pattern):** captures local texture patterns (leaf surface/vein roughness)
- **HOG (Histogram of Oriented Gradients):** captures shape/edge structure (leaf boundary and arrangement)

Combining color + texture + shape helps handle real-world variability better than any single descriptor.

### 4.2 Candidate classifiers
`src/train.py` compares classical models and hyperparameter settings:
- SVM (linear/RBF)
- Random Forest
- Gradient Boosting

Selection criterion:
1. Highest validation accuracy
2. Macro-F1 as tie-breaker

### 4.3 Reproducibility design
- Fixed random seed (`42`)
- Deterministic split pipeline
- Saved artifacts for exact reuse:
  - `models/best_model.pkl`
  - `models/label_encoder.pkl`
  - `outputs/training_summary.json`

### 4.4 How to run
```bash
python src/build_dataset.py
python src/train.py
```

---

## 5) Task 4 — Evaluation & Error Analysis

### 5.1 Required metrics and artifacts
`src/evaluate.py` is designed to produce:
- Overall test **accuracy**
- **Per-class** precision/recall/F1/support
- **Confusion matrix** image
- Misclassification table and optional copied error examples

Expected files:
- `outputs/metrics.json`
- `outputs/confusion_matrix.png`
- `outputs/error_cases.csv`
- `outputs/error_cases/` (optional, if `--copy-errors > 0`)

### 5.2 Evaluation command
```bash
python src/evaluate.py --copy-errors 20
```

### 5.3 Structured error analysis (feature-linked)
Typical failure modes and interpretation:

1. **Background bias**  
   - Symptom: model predicts class correlated with background context rather than plant itself.  
   - Feature link: color histograms may absorb non-plant background colors.

2. **Lighting shifts (illumination changes)**  
   - Symptom: performance drops under strong shadow/backlight/high exposure.  
   - Feature link: HSV/color histogram distributions shift with illumination.

3. **Similar-looking species confusion**  
   - Symptom: confusion between visually close classes (e.g., palm-like foliage).  
   - Feature link: handcrafted descriptors may lack fine semantic discriminability.

4. **Viewpoint/scale/blur sensitivity**  
   - Symptom: off-angle or far-distance samples are misclassified more often.  
   - Feature link: HOG and LBP can be sensitive to scale, blur, and strong perspective changes.

---

## 6) Task 5 — Application

A local Streamlit app is provided in `app/app.py`.

### 6.1 Functionality
- User uploads an image (`jpg/jpeg/png/bmp/webp`)
- App extracts the same handcrafted feature vector
- App outputs predicted class
- If model supports probability, app shows top-3 class probabilities

### 6.2 Run locally
```bash
streamlit run app/app.py
```

---

## 7) Task 6 — AI Collaboration Statement

### 7.1 How Generative AI assisted
Generative AI was used as a coding assistant for:
- Structuring training/evaluation scripts
- Clarifying scikit-learn workflow design
- Suggesting robust output artifact formats

### 7.2 Limitations observed
- Generated suggestions can miss environment constraints (e.g., unavailable packages)
- Hyperparameters suggested by AI are not always optimal for small datasets
- Some recommendations need adaptation to assignment restrictions (no deep learning)

### 7.3 Human verification and improvement
- The pipeline was constrained to classical ML only
- Feature extraction was kept consistent across train/eval/app via shared module
- Data validation and split logic were explicitly implemented for reproducibility
- Model selection criteria were made explicit and auditable via JSON summary

---

## 8) Reproducibility Checklist

- [x] Class folder dataset format documented  
- [x] Data validation and duplicate checks implemented  
- [x] Stratified train/val/test split implemented  
- [x] Traditional ML only (no deep learning)  
- [x] Artifact saving for model and label encoder  
- [x] Test metrics + confusion matrix outputs defined  
- [x] Local prediction app included

---

## 9) Commands Summary

```bash
# Install dependencies
pip install -r requirements.txt

# Build validated split
python src/build_dataset.py

# Train and select best model
python src/train.py

# Evaluate on held-out test set
python src/evaluate.py --copy-errors 20

# Launch local app
streamlit run app/app.py
```

---

## 10) Submission Notes
- Export this Markdown report to PDF with required naming format:  
  `Assignment2_Report_<your_ID>_<your_name>.pdf`
- Zip code + dataset with required naming format:  
  `Assignment2_Code_<your_ID>_<your_name>.zip`
- Submit both files to Blackboard.

