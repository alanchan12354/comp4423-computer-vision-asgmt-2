from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from features import extract_feature_vector


DEFAULT_MODEL_PATH = Path("models/best_model.pkl")
DEFAULT_LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image prediction using trained model.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model (.pkl).",
    )
    parser.add_argument(
        "--label-encoder",
        type=Path,
        default=DEFAULT_LABEL_ENCODER_PATH,
        help="Path to fitted label encoder (.pkl).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top probabilities to print when predict_proba is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists() or not args.image.is_file():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not args.model.exists() or not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not args.label_encoder.exists() or not args.label_encoder.is_file():
        raise FileNotFoundError(f"Label encoder file not found: {args.label_encoder}")
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0, got: {args.top_k}")

    model = joblib.load(args.model)
    label_encoder = joblib.load(args.label_encoder)

    feature_vector = extract_feature_vector(args.image)
    x = feature_vector.reshape(1, -1).astype(np.float32)

    pred_enc = model.predict(x)[0]
    pred_label = label_encoder.inverse_transform(np.asarray([pred_enc], dtype=int))[0]

    print(f"Predicted class: {pred_label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]

        if hasattr(model, "classes_"):
            enc_classes = np.asarray(model.classes_, dtype=int)
        else:
            enc_classes = np.arange(len(probabilities), dtype=int)

        class_labels = label_encoder.inverse_transform(enc_classes)

        class_prob_pairs = list(zip(class_labels, probabilities))
        class_prob_pairs.sort(key=lambda item: item[1], reverse=True)

        top_k = min(args.top_k, len(class_prob_pairs))
        print(f"Top-{top_k} probabilities:")
        for class_name, prob in class_prob_pairs[:top_k]:
            print(f"  {class_name}: {prob:.4f}")
    else:
        print("Model does not support predict_proba; skipping confidence/top-k output.")


if __name__ == "__main__":
    main()
