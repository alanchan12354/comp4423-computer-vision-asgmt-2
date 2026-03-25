from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from features import extract_feature_vector


SPLIT_PATH = Path("data/splits/split.csv")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"


def load_split(split_path: Path) -> pd.DataFrame:
    """Load split CSV and validate required columns."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    df = pd.read_csv(split_path)
    required_cols = {"filepath", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in split CSV: {sorted(missing)}")

    allowed_splits = {"train", "val", "test"}
    bad_splits = set(df["split"].unique()) - allowed_splits
    if bad_splits:
        raise ValueError(f"Unexpected split values found: {sorted(bad_splits)}")

    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract feature vectors for train and validation sets."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Both train and val splits must have at least one sample.")

    def _extract(filepaths: List[str]) -> np.ndarray:
        feats = [extract_feature_vector(path) for path in filepaths]
        return np.vstack(feats).astype(np.float32)

    x_train = _extract(train_df["filepath"].tolist())
    x_val = _extract(val_df["filepath"].tolist())

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    return x_train, y_train, x_val, y_val


def build_candidates(random_seed: int = 42) -> Dict[str, Tuple[Any, List[Dict[str, Any]]]]:
    """Define candidate models and manual hyperparameter grids."""
    candidates: Dict[str, Tuple[Any, List[Dict[str, Any]]]] = {
        "svm": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVC(random_state=random_seed)),
                ]
            ),
            [
                {"model__kernel": "linear", "model__C": 0.1},
                {"model__kernel": "linear", "model__C": 1.0},
                {"model__kernel": "rbf", "model__C": 1.0, "model__gamma": "scale"},
                {"model__kernel": "rbf", "model__C": 10.0, "model__gamma": "scale"},
            ],
        ),
        "random_forest": (
            RandomForestClassifier(random_state=random_seed),
            [
                {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
                {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
                {"n_estimators": 200, "max_depth": 20, "min_samples_split": 2},
                {"n_estimators": 300, "max_depth": 20, "min_samples_split": 4},
            ],
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_seed),
            [
                {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
                {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
                {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
                {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 2},
            ],
        ),
    }
    return candidates


def evaluate_candidates(
    x_train: np.ndarray,
    y_train_enc: np.ndarray,
    x_val: np.ndarray,
    y_val_enc: np.ndarray,
) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    """Train and validate all candidate hyperparameter configurations."""
    candidates = build_candidates(random_seed=42)

    best_model = None
    best_result: Dict[str, Any] = {}
    all_results: List[Dict[str, Any]] = []

    for model_name, (base_estimator, param_grid) in candidates.items():
        for params in param_grid:
            estimator = clone(base_estimator)
            estimator.set_params(**params)
            estimator.fit(x_train, y_train_enc)

            val_pred = estimator.predict(x_val)
            val_accuracy = float(accuracy_score(y_val_enc, val_pred))
            val_f1_macro = float(f1_score(y_val_enc, val_pred, average="macro"))

            result = {
                "model_name": model_name,
                "params": params,
                "val_accuracy": val_accuracy,
                "val_f1_macro": val_f1_macro,
            }
            all_results.append(result)

            is_better = (
                not best_result
                or val_accuracy > best_result["val_accuracy"]
                or (
                    np.isclose(val_accuracy, best_result["val_accuracy"])
                    and val_f1_macro > best_result["val_f1_macro"]
                )
            )
            if is_better:
                best_model = estimator
                best_result = result

    if best_model is None:
        raise RuntimeError("No candidate model was successfully trained.")

    return best_model, best_result, all_results


def save_artifacts(model: Any, label_encoder: LabelEncoder, summary: Dict[str, Any]) -> None:
    """Persist trained model, label encoder, and training summary."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, BEST_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    split_df = load_split(SPLIT_PATH)
    x_train, y_train_raw, x_val, y_val_raw = build_feature_matrix(split_df)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_raw)
    y_val_enc = label_encoder.transform(y_val_raw)

    best_model, best_result, all_results = evaluate_candidates(
        x_train=x_train,
        y_train_enc=y_train_enc,
        x_val=x_val,
        y_val_enc=y_val_enc,
    )

    summary = {
        "split_csv": str(SPLIT_PATH),
        "n_train": int(len(y_train_raw)),
        "n_val": int(len(y_val_raw)),
        "n_features": int(x_train.shape[1]),
        "classes": label_encoder.classes_.tolist(),
        "best_model": best_result,
        "candidate_results": all_results,
        "artifacts": {
            "model_path": str(BEST_MODEL_PATH),
            "label_encoder_path": str(LABEL_ENCODER_PATH),
            "summary_path": str(SUMMARY_PATH),
        },
    }

    save_artifacts(best_model, label_encoder, summary)

    print("Training complete.")
    print(f"Best model: {best_result['model_name']}")
    print(f"Best params: {best_result['params']}")
    print(f"Validation accuracy: {best_result['val_accuracy']:.4f}")
    print(f"Validation macro-F1: {best_result['val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
