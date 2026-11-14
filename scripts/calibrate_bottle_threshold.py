from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split

from src.visioninspect.inference import (
    VisionInspectInference,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    "models/patchcore_bottle_memory.pt"
)

TRAIN_GOOD_DIR = Path(
    "data/bottle/train/good"
)

SEED = 42


# ============================================================
# GET TRAIN IMAGE PATHS
# ============================================================

image_paths = sorted(
    TRAIN_GOOD_DIR.glob("*.png")
)

total_images = len(
    image_paths
)

train_size = int(
    total_images * 0.90
)

val_size = (
    total_images - train_size
)


# ============================================================
# REPRODUCE ORIGINAL 90/10 SPLIT
# ============================================================

generator = torch.Generator().manual_seed(
    SEED
)

train_subset, val_subset = random_split(
    image_paths,
    [
        train_size,
        val_size,
    ],
    generator=generator,
)

validation_paths = [
    image_paths[index]
    for index in val_subset.indices
]


print(
    f"Total normal training images: {total_images}"
)

print(
    f"Training subset: {train_size}"
)

print(
    f"Validation subset: {val_size}"
)


# ============================================================
# LOAD MODEL
# ============================================================

engine = VisionInspectInference(
    model_path=MODEL_PATH,
    threshold=0.0,
    device="cpu",
)


# ============================================================
# SCORE VALIDATION NORMALS
# ============================================================

scores = []

print(
    "\n=============================="
)

print(
    "NORMAL VALIDATION SCORES"
)

print(
    "==============================\n"
)


for image_path in validation_paths:

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    result = engine.predict_image(
        image
    )

    score = result[
        "score"
    ]

    scores.append(
        score
    )

    print(
        f"{image_path.name:15s} "
        f"{score:.6f}"
    )


scores = np.array(
    scores
)


# ============================================================
# THRESHOLD CANDIDATES
# ============================================================

threshold_95 = np.percentile(
    scores,
    95
)

threshold_99 = np.percentile(
    scores,
    99
)

threshold_max = scores.max()

threshold_mean_3std = (
    scores.mean()
    +
    3 * scores.std()
)


# ============================================================
# RESULTS
# ============================================================

print(
    "\n=============================="
)

print(
    "CALIBRATION RESULTS"
)

print(
    "=============================="
)


print(
    f"\nMean:              "
    f"{scores.mean():.6f}"
)

print(
    f"Std:               "
    f"{scores.std():.6f}"
)

print(
    f"Minimum:           "
    f"{scores.min():.6f}"
)

print(
    f"Maximum:           "
    f"{scores.max():.6f}"
)


print(
    "\nTHRESHOLD CANDIDATES"
)

print(
    f"95th percentile:   "
    f"{threshold_95:.6f}"
)

print(
    f"99th percentile:   "
    f"{threshold_99:.6f}"
)

print(
    f"Maximum validation:"
    f" {threshold_max:.6f}"
)

print(
    f"Mean + 3 std:      "
    f"{threshold_mean_3std:.6f}"
)


print(
    "\nRecommended candidate:"
)

print(
    f"{threshold_99:.6f}"
)

print(
    "\nThis threshold uses only normal "
    "validation data — not test labels."
)