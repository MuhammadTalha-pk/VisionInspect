from pathlib import Path

import numpy as np
from PIL import Image

from src.visioninspect.inference import (
    VisionInspectInference,
)


MODEL_PATH = (
    "models/patchcore_bottle_memory.pt"
)

CURRENT_THRESHOLD = (
    0.59880111515522
)

GOOD_FOLDER = Path(
    "data/bottle/test/good"
)

TEST_ROOT = Path(
    "data/bottle/test"
)


engine = VisionInspectInference(
    model_path=MODEL_PATH,
    threshold=CURRENT_THRESHOLD,
    device="cpu",
)


def score_image(
    image_path,
):
    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    result = engine.predict_image(
        image
    )

    return result[
        "score"
    ]


# ============================================================
# SCORE GOOD IMAGES
# ============================================================

good_scores = []

print(
    "\n=============================="
)

print(
    "GOOD IMAGE SCORES"
)

print(
    "==============================\n"
)


good_paths = sorted(
    GOOD_FOLDER.glob(
        "*.png"
    )
)


for image_path in good_paths:

    score = score_image(
        image_path
    )

    good_scores.append(
        score
    )

    status = (
        "FALSE POSITIVE"
        if score >= CURRENT_THRESHOLD
        else "NORMAL"
    )

    print(
        f"{image_path.name:15s} "
        f"{score:.6f}   "
        f"{status}"
    )


# ============================================================
# SCORE DEFECTIVE IMAGES
# ============================================================

defect_scores = []

defect_records = []


for defect_folder in sorted(
    TEST_ROOT.iterdir()
):

    if (
        not defect_folder.is_dir()
        or defect_folder.name == "good"
    ):
        continue

    for image_path in sorted(
        defect_folder.glob(
            "*.png"
        )
    ):

        score = score_image(
            image_path
        )

        defect_scores.append(
            score
        )

        defect_records.append(
            (
                defect_folder.name,
                image_path.name,
                score,
            )
        )


good_scores = np.array(
    good_scores
)

defect_scores = np.array(
    defect_scores
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n=============================="
)

print(
    "THRESHOLD ANALYSIS"
)

print(
    "=============================="
)


print(
    f"\nCurrent threshold: "
    f"{CURRENT_THRESHOLD:.6f}"
)


print(
    "\nGOOD IMAGES"
)

print(
    f"Count: {len(good_scores)}"
)

print(
    f"Minimum: {good_scores.min():.6f}"
)

print(
    f"Mean: {good_scores.mean():.6f}"
)

print(
    f"Median: {np.median(good_scores):.6f}"
)

print(
    f"Maximum: {good_scores.max():.6f}"
)


false_positives = (
    good_scores
    >= CURRENT_THRESHOLD
).sum()


print(
    f"False positives: "
    f"{false_positives}"
)


print(
    f"False positive rate: "
    f"{false_positives / len(good_scores):.2%}"
)


print(
    "\nDEFECTIVE IMAGES"
)

print(
    f"Count: {len(defect_scores)}"
)

print(
    f"Minimum: {defect_scores.min():.6f}"
)

print(
    f"Mean: {defect_scores.mean():.6f}"
)

print(
    f"Maximum: {defect_scores.max():.6f}"
)


false_negatives = (
    defect_scores
    < CURRENT_THRESHOLD
).sum()


print(
    f"False negatives: "
    f"{false_negatives}"
)


# ============================================================
# SEPARATION
# ============================================================

max_good = good_scores.max()

min_defect = defect_scores.min()


print(
    "\n=============================="
)

print(
    "SCORE SEPARATION"
)

print(
    "=============================="
)


print(
    f"\nHighest normal score: "
    f"{max_good:.6f}"
)

print(
    f"Lowest defect score: "
    f"{min_defect:.6f}"
)


gap = (
    min_defect
    - max_good
)


print(
    f"Gap: {gap:.6f}"
)


if gap > 0:

    midpoint = (
        max_good
        + min_defect
    ) / 2

    print(
        "\nThe model separates the Bottle "
        "test scores completely."
    )

    print(
        f"Diagnostic midpoint: "
        f"{midpoint:.6f}"
    )

else:

    print(
        "\nNormal and defective image "
        "score distributions overlap."
    )


print(
    "\nIMPORTANT:"
)

print(
    "The diagnostic midpoint uses test labels."
)

print(
    "It must NOT be reported as an unbiased "
    "benchmark threshold."
)