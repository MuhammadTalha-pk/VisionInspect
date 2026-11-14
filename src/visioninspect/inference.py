from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from .preprocessing import (
    preprocess_image,
    denormalize_image,
)

from .patchcore import PatchCoreDetector


class VisionInspectInference:

    def __init__(
        self,
        model_path,
        threshold,
        device=None,
    ):
        self.threshold = float(threshold)

        self.detector = PatchCoreDetector(
            model_path=model_path,
            device=device,
        )

    @staticmethod
    def upscale_anomaly_map(
        anomaly_map,
        output_size=256,
    ):
        """
        Convert PatchCore 32x32 map into 256x256 map.
        """

        if anomaly_map.dim() == 3:
            anomaly_map = anomaly_map.unsqueeze(1)

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )

        anomaly_map = F.avg_pool2d(
            anomaly_map,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        return anomaly_map

    @staticmethod
    def normalize_heatmap(
        anomaly_map,
    ):
        """
        Normalize anomaly map to [0, 1]
        for visualization only.
        """

        minimum = anomaly_map.min()
        maximum = anomaly_map.max()

        normalized = (
            anomaly_map - minimum
        ) / (
            maximum - minimum + 1e-8
        )

        return normalized

    def predict_image(
        self,
        image,
    ):
        """
        Run VisionInspect on a PIL image.

        Returns:
            dict containing:
            prediction
            score
            threshold
            heatmap
            image
        """

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        tensor = preprocess_image(
            image
        )

        scores, anomaly_maps = (
            self.detector.predict(
                tensor
            )
        )

        score = float(
            scores[0].item()
        )

        prediction = (
            "ANOMALOUS"
            if score >= self.threshold
            else "NORMAL"
        )

        anomaly_map = self.upscale_anomaly_map(
            anomaly_maps,
            output_size=256,
        )

        normalized_map = self.normalize_heatmap(
            anomaly_map
        )

        heatmap = (
            normalized_map[0, 0]
            .cpu()
            .numpy()
        )

        display_image = denormalize_image(
            tensor
        )

        display_image = (
            display_image
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        return {
            "prediction": prediction,
            "score": score,
            "threshold": self.threshold,
            "heatmap": heatmap,
            "image": display_image,
        }