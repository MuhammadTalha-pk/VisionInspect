import torch

from torchvision.models import (
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
)

from torchvision.models.feature_extraction import (
    create_feature_extractor,
)


class VisionFeatureExtractor:

    def __init__(self, device=None):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        weights = Wide_ResNet50_2_Weights.DEFAULT

        backbone = wide_resnet50_2(
            weights=weights
        )

        backbone.eval()

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "layer2",
                "layer3": "layer3",
            }
        )

        self.extractor = (
            self.extractor
            .to(self.device)
            .eval()
        )


    @torch.no_grad()
    def extract(self, images):

        images = images.to(self.device)

        features = self.extractor(
            images
        )

        return features