from pathlib import Path

import torch
import torch.nn.functional as F

from .feature_extractor import VisionFeatureExtractor


class PatchCoreDetector:

    def __init__(
        self,
        model_path,
        device=None,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        # Load frozen pretrained CNN feature extractor
        self.feature_extractor = VisionFeatureExtractor(
            device=self.device
        )

        # Load PatchCore memory bank created in Colab
        checkpoint = torch.load(
            Path(model_path),
            map_location="cpu"
        )

        self.memory_bank = checkpoint[
            "memory_bank"
        ].to(self.device)

        self.projection_matrix = checkpoint[
            "projection_matrix"
        ].to(self.device)

        self.embedding_dim = checkpoint[
            "embedding_dim"
        ]

        self.coreset_size = checkpoint[
            "coreset_size"
        ]

    @torch.no_grad()
    def extract_patch_embeddings(
        self,
        images,
    ):
        """
        Extract multi-scale CNN features from layer2
        and layer3 and convert them into local patches.
        """

        features = self.feature_extractor.extract(
            images
        )

        layer2 = features["layer2"]
        layer3 = features["layer3"]

        # layer2: [B, 512, 32, 32]
        # layer3: [B, 1024, 16, 16]

        # Resize layer3 so spatial dimensions match layer2
        layer3 = F.interpolate(
            layer3,
            size=layer2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Combine layer2 + layer3
        # 512 + 1024 = 1536 features
        combined = torch.cat(
            [layer2, layer3],
            dim=1
        )

        batch_size, channels, height, width = (
            combined.shape
        )

        # [B, 1536, 32, 32]
        #
        # becomes
        #
        # [B, 1024, 1536]

        patches = (
            combined
            .permute(0, 2, 3, 1)
            .reshape(
                batch_size,
                height * width,
                channels
            )
        )

        # Normalize each patch feature vector
        patches = F.normalize(
            patches,
            p=2,
            dim=-1
        )

        return patches, height, width

    @torch.no_grad()
    def project_embeddings(
        self,
        patches,
    ):
        """
        Project 1536-dimensional patch embeddings
        into the same 128-dimensional feature space
        used when the memory bank was created.
        """

        projected = torch.matmul(
            patches,
            self.projection_matrix
        )

        projected = F.normalize(
            projected,
            p=2,
            dim=-1
        )

        return projected

    @torch.no_grad()
    def predict(
        self,
        images,
    ):
        """
        Calculate image-level anomaly scores and
        spatial PatchCore anomaly maps.

        Returns:
            image_scores:
                Tensor [B]

            anomaly_maps:
                Tensor [B, 32, 32]
        """

        images = images.to(
            self.device
        )

        # -----------------------------------------
        # Extract local CNN features
        # -----------------------------------------

        patches, height, width = (
            self.extract_patch_embeddings(
                images
            )
        )

        # -----------------------------------------
        # Project 1536 -> 128
        # -----------------------------------------

        patches = self.project_embeddings(
            patches
        )

        batch_size = patches.shape[0]

        # [B, 1024, 128]
        # ->
        # [B * 1024, 128]

        flat_patches = patches.reshape(
            -1,
            self.embedding_dim
        )

        # -----------------------------------------
        # Nearest-neighbour anomaly scoring
        # -----------------------------------------

        distances = torch.cdist(
            flat_patches,
            self.memory_bank
        )

        nearest_distances = (
            distances
            .min(dim=1)
            .values
        )

        # Restore spatial patch layout
        anomaly_maps = nearest_distances.reshape(
            batch_size,
            height,
            width
        )

        # -----------------------------------------
        # Image-level anomaly score
        # -----------------------------------------

        flat_maps = anomaly_maps.flatten(
            start_dim=1
        )

        # Mean of top 1% most anomalous patches
        k = max(
            1,
            int(
                flat_maps.shape[1]
                * 0.01
            )
        )

        image_scores = torch.topk(
            flat_maps,
            k=k,
            dim=1
        ).values.mean(
            dim=1
        )

        return (
            image_scores.cpu(),
            anomaly_maps.cpu()
        )