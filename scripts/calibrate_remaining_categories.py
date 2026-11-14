from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split

from src.visioninspect.inference import VisionInspectInference


# ============================================================
# CONFIG
# ============================================================

SEED = 42

CATEGORIES = {

    "hazelnut": {
        "model_path":
            "models/patchcore_hazelnut_memory.pt",

        "train_good":
            "data/hazelnut/train/good",

        "test_root":
            "data/hazelnut/test",
    },

    "tile": {
        "model_path":
            "models/patchcore_tile_memory.pt",

        "train_good":
            "data/tile/train/good",

        "test_root":
            "data/tile/test",
    },
}


# ============================================================
# SCORE ONE IMAGE
# ============================================================

def score_image(
    engine,
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

    return float(
        result["score"]
    )


# ============================================================
# CALIBRATE CATEGORY
# ============================================================

def calibrate_category(
    category,
    config,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"{category.upper()} CALIBRATION"
    )

    print(
        "=" * 60
    )


    train_good_dir = Path(
        config["train_good"]
    )

    test_root = Path(
        config["test_root"]
    )


    # --------------------------------------------------------
    # NORMAL TRAINING IMAGES
    # --------------------------------------------------------

    image_paths = sorted(
        train_good_dir.glob(
            "*.png"
        )
    )

    total_images = len(
        image_paths
    )

    train_size = int(
        total_images * 0.90
    )

    val_size = (
        total_images
        - train_size
    )


    print(
        f"\nTotal normal training images: "
        f"{total_images}"
    )

    print(
        f"Training subset: {train_size}"
    )

    print(
        f"Validation subset: {val_size}"
    )


    # --------------------------------------------------------
    # REPRODUCE 90/10 SPLIT
    # --------------------------------------------------------

    generator = (
        torch.Generator()
        .manual_seed(SEED)
    )

    train_subset, val_subset = (
        random_split(
            image_paths,
            [
                train_size,
                val_size,
            ],
            generator=generator,
        )
    )

    validation_paths = [
        image_paths[index]
        for index
        in val_subset.indices
    ]


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    engine = VisionInspectInference(
        model_path=config[
            "model_path"
        ],
        threshold=0.0,
        device="cpu",
    )


    # --------------------------------------------------------
    # VALIDATION SCORES
    # --------------------------------------------------------

    validation_scores = []

    print(
        "\nNORMAL VALIDATION SCORES\n"
    )


    for image_path in validation_paths:

        score = score_image(
            engine,
            image_path,
        )

        validation_scores.append(
            score
        )

        print(
            f"{image_path.name:15s} "
            f"{score:.6f}"
        )


    validation_scores = np.array(
        validation_scores
    )


    # --------------------------------------------------------
    # THRESHOLD CANDIDATES
    # --------------------------------------------------------

    mean_score = (
        validation_scores.mean()
    )

    std_score = (
        validation_scores.std()
    )

    threshold_95 = np.percentile(
        validation_scores,
        95,
    )

    threshold_99 = np.percentile(
        validation_scores,
        99,
    )

    threshold_max = (
        validation_scores.max()
    )

    threshold_3std = (
        mean_score
        +
        3 * std_score
    )


    # --------------------------------------------------------
    # CALIBRATION RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CALIBRATION RESULTS"
    )

    print(
        "=" * 60
    )


    print(
        f"\nMean:               "
        f"{mean_score:.6f}"
    )

    print(
        f"Std:                "
        f"{std_score:.6f}"
    )

    print(
        f"Minimum:            "
        f"{validation_scores.min():.6f}"
    )

    print(
        f"Maximum:            "
        f"{validation_scores.max():.6f}"
    )


    print(
        "\nTHRESHOLD CANDIDATES"
    )

    print(
        f"95th percentile:    "
        f"{threshold_95:.6f}"
    )

    print(
        f"99th percentile:    "
        f"{threshold_99:.6f}"
    )

    print(
        f"Maximum validation: "
        f"{threshold_max:.6f}"
    )

    print(
        f"Mean + 3 std:       "
        f"{threshold_3std:.6f}"
    )


    # --------------------------------------------------------
    # TEST DIAGNOSTIC
    # --------------------------------------------------------
    # IMPORTANT:
    # Test labels are only inspected to understand separation.
    # They are NOT used to calculate the deployment threshold.
    # --------------------------------------------------------

    good_scores = []

    defect_scores = []


    good_folder = (
        test_root / "good"
    )


    for image_path in sorted(
        good_folder.glob(
            "*.png"
        )
    ):

        score = score_image(
            engine,
            image_path,
        )

        good_scores.append(
            score
        )


    for defect_folder in sorted(
        test_root.iterdir()
    ):

        if (
            not defect_folder.is_dir()
            or defect_folder.name
            == "good"
        ):
            continue

        for image_path in sorted(
            defect_folder.glob(
                "*.png"
            )
        ):

            score = score_image(
                engine,
                image_path,
            )

            defect_scores.append(
                score
            )


    good_scores = np.array(
        good_scores
    )

    defect_scores = np.array(
        defect_scores
    )


    # --------------------------------------------------------
    # DIAGNOSTIC RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TEST SCORE DIAGNOSTIC"
    )

    print(
        "=" * 60
    )


    print(
        f"\nNormal test images: "
        f"{len(good_scores)}"
    )

    print(
        f"Highest normal score: "
        f"{good_scores.max():.6f}"
    )

    print(
        f"Lowest normal score:  "
        f"{good_scores.min():.6f}"
    )


    print(
        f"\nDefective test images: "
        f"{len(defect_scores)}"
    )

    print(
        f"Lowest defect score:  "
        f"{defect_scores.min():.6f}"
    )

    print(
        f"Highest defect score: "
        f"{defect_scores.max():.6f}"
    )


    separation_gap = (
        defect_scores.min()
        -
        good_scores.max()
    )


    print(
        f"\nSeparation gap: "
        f"{separation_gap:.6f}"
    )


    if separation_gap > 0:

        print(
            "Normal and defective "
            "test scores do not overlap."
        )

    else:

        print(
            "Normal and defective "
            "test score distributions overlap."
        )


    # --------------------------------------------------------
    # EVALUATE VALIDATION-ONLY 3STD THRESHOLD
    # --------------------------------------------------------

    false_positives = (
        good_scores
        >= threshold_3std
    ).sum()

    false_negatives = (
        defect_scores
        < threshold_3std
    ).sum()


    print(
        "\nMEAN + 3 STD THRESHOLD TEST"
    )

    print(
        f"Threshold: "
        f"{threshold_3std:.6f}"
    )

    print(
        f"False positives: "
        f"{false_positives}"
    )

    print(
        f"False negatives: "
        f"{false_negatives}"
    )


    print(
        "\nRecommended validation-only "
        "candidate:"
    )

    print(
        f"{threshold_3std:.6f}"
    )


    return {
        "category":
            category,

        "threshold":
            threshold_3std,

        "max_normal_test":
            good_scores.max(),

        "min_defect_test":
            defect_scores.min(),

        "false_positives":
            int(false_positives),

        "false_negatives":
            int(false_negatives),
    }


# ============================================================
# RUN ALL
# ============================================================

results = []


for category, config in (
    CATEGORIES.items()
):

    result = calibrate_category(
        category,
        config,
    )

    results.append(
        result
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n\n"
    + "=" * 60
)

print(
    "FINAL CALIBRATION SUMMARY"
)

print(
    "=" * 60
)


for result in results:

    print(
        f"\n{result['category'].upper()}"
    )

    print(
        f"Validation threshold: "
        f"{result['threshold']:.6f}"
    )

    print(
        f"Highest normal test:  "
        f"{result['max_normal_test']:.6f}"
    )

    print(
        f"Lowest defect test:   "
        f"{result['min_defect_test']:.6f}"
    )

    print(
        f"False positives:      "
        f"{result['false_positives']}"
    )

    print(
        f"False negatives:      "
        f"{result['false_negatives']}"
    )


print(
    "\nNOTE:"
)

print(
    "Thresholds above are calculated only "
    "from normal validation statistics."
)

print(
    "Test labels are shown only for diagnostic "
    "evaluation and are not used to calculate "
    "the thresholds."
)