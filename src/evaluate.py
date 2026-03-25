from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from features import extract_feature_vector


SPLIT_PATH = Path("data/splits/split.csv")
MODEL_PATH = Path("models/best_model.pkl")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")
OUTPUT_DIR = Path("outputs")
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
ERROR_CASES_CSV_PATH = OUTPUT_DIR / "error_cases.csv"
ERROR_CASES_DIR = OUTPUT_DIR / "error_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on the test split.")
    parser.add_argument("--split-csv", type=Path, default=SPLIT_PATH, help="Path to split CSV.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to trained model file.")
    parser.add_argument(
        "--label-encoder",
        type=Path,
        default=LABEL_ENCODER_PATH,
        help="Path to fitted label encoder file.",
    )
    parser.add_argument(
        "--copy-errors",
        type=int,
        default=0,
        help="Optionally copy up to this many misclassified images into outputs/error_cases/.",
    )
    return parser.parse_args()


def load_test_split(split_csv: Path) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    required_cols = {"filepath", "label", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV missing required columns: {sorted(missing)}")

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if test_df.empty:
        raise ValueError("No samples found for split='test'.")

    return test_df[["filepath", "label"]]


def build_features(filepaths: List[str]) -> np.ndarray:
    features = [extract_feature_vector(path) for path in filepaths]
    return np.vstack(features).astype(np.float32)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    accuracy = float(accuracy_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0,
    )

    per_class = {
        cls: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(class_names, precision, recall, f1, support)
    }

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
    }


def save_confusion_matrix_figure(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Test Split)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_error_cases(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    copy_errors: int,
) -> Tuple[pd.DataFrame, int]:
    pred_series = pd.Series(y_pred, name="pred_label")
    combined = pd.concat([test_df, pred_series], axis=1)

    errors_df = combined[combined["label"] != combined["pred_label"]].copy()
    errors_df = errors_df.rename(columns={"label": "true_label"})
    errors_df.to_csv(ERROR_CASES_CSV_PATH, index=False)

    copied = 0
    if copy_errors > 0 and not errors_df.empty:
        ERROR_CASES_DIR.mkdir(parents=True, exist_ok=True)
        for _, row in errors_df.head(copy_errors).iterrows():
            src = Path(row["filepath"])
            if not src.exists():
                continue
            safe_true = str(row["true_label"]).replace("/", "_")
            safe_pred = str(row["pred_label"]).replace("/", "_")
            dst_name = f"true-{safe_true}__pred-{safe_pred}__{src.name}"
            shutil.copy2(src, ERROR_CASES_DIR / dst_name)
            copied += 1

    return errors_df, copied


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.label_encoder.exists():
        raise FileNotFoundError(f"Label encoder file not found: {args.label_encoder}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = load_test_split(args.split_csv)
    x_test = build_features(test_df["filepath"].tolist())
    y_test = test_df["label"].to_numpy()

    model = joblib.load(args.model)
    label_encoder = joblib.load(args.label_encoder)

    y_pred_enc = model.predict(x_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    class_names = label_encoder.classes_.tolist()
    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, class_names=class_names)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_array = np.asarray(metrics["confusion_matrix"]["matrix"])
    save_confusion_matrix_figure(cm_array, class_names=class_names, out_path=CONFUSION_MATRIX_PATH)

    errors_df, copied = export_error_cases(test_df=test_df, y_pred=y_pred, copy_errors=args.copy_errors)

    print("Evaluation complete.")
    print(f"Test samples: {len(test_df)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PATH}")
    print(f"Saved error cases CSV to: {ERROR_CASES_CSV_PATH} ({len(errors_df)} rows)")
    if args.copy_errors > 0:
        print(f"Copied {copied} misclassified samples to: {ERROR_CASES_DIR}")


if __name__ == "__main__":
    main()
