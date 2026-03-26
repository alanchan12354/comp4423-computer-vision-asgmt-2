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

    st.write("Upload an image and run inference with the trained model.")

    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    label_encoder_path = st.text_input("Label encoder path", value=str(DEFAULT_LABEL_ENCODER_PATH))

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is None:
        st.info("Please upload an image to continue.")
        return

    try:
        image_bgr = decode_uploaded_image(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return

    preview_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.image(preview_rgb, caption="Image preview", use_container_width=True)

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
            feature_vector = extract_feature_vector(image_bgr)
            x = feature_vector.reshape(1, -1).astype(np.float32)

            pred_enc = model.predict(x)[0]
            pred_label = label_encoder.inverse_transform(np.asarray([pred_enc], dtype=int))[0]
            st.success(f"Predicted class: {pred_label}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x)[0]

                if hasattr(model, "classes_"):
                    enc_classes = np.asarray(model.classes_, dtype=int)
                else:
                    enc_classes = np.arange(len(probabilities), dtype=int)

                class_labels = label_encoder.inverse_transform(enc_classes)
                class_prob_pairs = list(zip(class_labels, probabilities))
                class_prob_pairs.sort(key=lambda item: item[1], reverse=True)

                top_3 = class_prob_pairs[:3]
                st.write("Top-3 probabilities:")
                for class_name, prob in top_3:
                    st.write(f"- **{class_name}**: {prob:.4f}")
            else:
                st.info("Model does not support predict_proba.")

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
