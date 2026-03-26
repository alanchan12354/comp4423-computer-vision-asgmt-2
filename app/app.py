from __future__ import annotations

from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_ENCODER_PATH, MODEL_PATH
from src.features import extract_feature_vector


DEFAULT_MODEL_PATH = Path(MODEL_PATH)
DEFAULT_LABEL_ENCODER_PATH = Path(LABEL_ENCODER_PATH)


@st.cache_resource
def load_artifacts(model_path: str, label_encoder_path: str):
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, label_encoder


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode uploaded image.")
    return image_bgr


def main() -> None:
    st.set_page_config(page_title="Leaf Classifier", page_icon="🌿")
    st.title("Leaf Image Classifier")

    st.write("Upload one or more images and run inference with the trained model.")

    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    label_encoder_path = st.text_input("Label encoder path", value=str(DEFAULT_LABEL_ENCODER_PATH))

    uploaded_files = st.file_uploader(
        "Choose image(s)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload one or more images to continue.")
        return

    decoded_images: list[tuple[str, np.ndarray]] = []
    decode_errors: list[str] = []
    for uploaded_file in uploaded_files:
        try:
            image_bgr = decode_uploaded_image(uploaded_file)
            decoded_images.append((uploaded_file.name, image_bgr))
        except Exception as exc:
            decode_errors.append(f"{uploaded_file.name}: {exc}")

    if decode_errors:
        st.error("Some files could not be read:")
        for error in decode_errors:
            st.write(f"- {error}")

    if not decoded_images:
        return

    st.write("Image previews:")
    for filename, image_bgr in decoded_images:
        preview_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, caption=filename, use_container_width=True)

    if st.button("Predict", type="primary"):
        model_file = Path(model_path)
        encoder_file = Path(label_encoder_path)

        if not model_file.exists():
            st.error(f"Model not found: {model_file}")
            return
        if not encoder_file.exists():
            st.error(f"Label encoder not found: {encoder_file}")
            return

        try:
            model, label_encoder = load_artifacts(str(model_file), str(encoder_file))
            features = [extract_feature_vector(image_bgr) for _, image_bgr in decoded_images]
            x = np.vstack(features).astype(np.float32)

            pred_encs = model.predict(x)
            pred_labels = label_encoder.inverse_transform(np.asarray(pred_encs, dtype=int))

            results = []
            for (filename, _), pred_label in zip(decoded_images, pred_labels):
                results.append({"filename": filename, "predicted_class": pred_label})

            st.success("Predictions completed.")
            st.write("Predicted classes:")
            st.table(results)

            if hasattr(model, "predict_proba"):
                probabilities_batch = model.predict_proba(x)

                if hasattr(model, "classes_"):
                    enc_classes = np.asarray(model.classes_, dtype=int)
                else:
                    enc_classes = np.arange(probabilities_batch.shape[1], dtype=int)

                class_labels = label_encoder.inverse_transform(enc_classes)
                st.write("Top-3 probabilities per image:")
                for (filename, _), probabilities in zip(decoded_images, probabilities_batch):
                    class_prob_pairs = list(zip(class_labels, probabilities))
                    class_prob_pairs.sort(key=lambda item: item[1], reverse=True)
                    top_3 = class_prob_pairs[:3]

                    st.write(f"**{filename}**")
                    for class_name, prob in top_3:
                        st.write(f"- **{class_name}**: {prob:.4f}")
            else:
                st.info("Model does not support predict_proba.")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
