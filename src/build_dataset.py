from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def can_decode_image(path: Path) -> bool:
    """Return True if OpenCV can decode the image file."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("Failed to read %s (%s)", path, exc)
        return False

    if not data:
        logger.warning("Skipping empty file: %s", path)
        return False

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        logger.warning("Skipping corrupted/unreadable image: %s", path)
        return False

    return True


def compute_sha1(path: Path) -> str:
    """Compute SHA-1 hash of a file."""
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(image_root: Path) -> pd.DataFrame:
    """Build dataset table with filepath and label, skipping unreadable images."""
    records: List[Dict[str, str]] = []
    hash_to_files: Dict[str, List[str]] = {}

    class_dirs = sorted([p for p in image_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under: {image_root}")

    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in sorted(class_dir.glob("*")):
            if not img_path.is_file():
                continue

            if not can_decode_image(img_path):
                continue

            file_hash = compute_sha1(img_path)
            rel_path = img_path.as_posix()
            hash_to_files.setdefault(file_hash, []).append(rel_path)
            records.append({"filepath": rel_path, "label": label, "sha1": file_hash})

    duplicates = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}
    if duplicates:
        logger.warning("Potential duplicates found: %d hash groups", len(duplicates))
        for h, paths in duplicates.items():
            logger.warning("SHA1 %s appears in %d files", h, len(paths))
            for p in paths:
                logger.warning("  - %s", p)
    else:
        logger.info("No duplicates detected by SHA-1.")

    if not records:
        raise RuntimeError("No valid images found after validation.")

    return pd.DataFrame.from_records(records)


def create_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create stratified train/val/test split with approximately 70/15/15 ratio."""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"],
    )

    split_train = train_df.assign(split="train")
    split_val = val_df.assign(split="val")
    split_test = test_df.assign(split="test")

    split_df = (
        pd.concat([split_train, split_val, split_test], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return split_df


def summarize_and_save_counts(split_df: pd.DataFrame, output_dir: Path) -> None:
    """Print and save per-class counts for full data and each split."""
    full_counts = split_df.groupby("label").size().rename("count").reset_index()
    full_counts.insert(0, "split", "full")

    split_counts = (
        split_df.groupby(["split", "label"]).size().rename("count").reset_index()
    )

    counts_df = pd.concat([full_counts, split_counts], ignore_index=True)
    counts_path = output_dir / "class_counts.csv"
    counts_df.to_csv(counts_path, index=False)

    print("\nPer-class counts (full set):")
    print(full_counts[["label", "count"]].sort_values("label").to_string(index=False))

    print("\nPer-class counts by split:")
    for split_name in ["train", "val", "test"]:
        current = split_counts[split_counts["split"] == split_name]
        print(f"\n[{split_name}]")
        print(current[["label", "count"]].sort_values("label").to_string(index=False))

    logger.info("Saved class counts to %s", counts_path)


def main() -> None:
    image_root = Path("img-train")
    output_dir = Path("data/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(image_root)
    split_df = create_splits(df, seed=42)

    split_path = output_dir / "split.csv"
    split_df[["filepath", "label", "split"]].to_csv(split_path, index=False)
    logger.info("Saved split file to %s", split_path)

    summarize_and_save_counts(split_df, output_dir)


if __name__ == "__main__":
    main()
