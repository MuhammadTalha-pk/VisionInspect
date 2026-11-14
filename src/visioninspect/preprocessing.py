from PIL import Image
import torch
from torchvision import transforms


IMAGE_SIZE = 256

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into the normalized tensor
    expected by VisionInspect.

    Returns:
        Tensor with shape [1, 3, 256, 256]
    """

    image = image.convert("RGB")

    tensor = image_transform(image)

    return tensor.unsqueeze(0)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert an ImageNet-normalized tensor back into
    displayable RGB values in the range [0, 1].
    """

    if tensor.dim() == 4:
        tensor = tensor[0]

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=tensor.dtype,
        device=tensor.device
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGENET_STD,
        dtype=tensor.dtype,
        device=tensor.device
    ).view(3, 1, 1)

    image = tensor * std + mean

    return image.clamp(0, 1)