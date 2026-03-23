"""
Dataset and augmentation utilities for self-supervised learning.
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image


def _ensure_tuple(value, channels):
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    return tuple(float(value) for _ in range(channels))


class ToFloatTensor:
    """Convert PIL/NumPy/tensor images to float tensors in CHW format."""

    def __call__(self, img):
        if isinstance(img, Image.Image):
            return TF.to_tensor(img)

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = img[None, :, :]
            elif img.ndim == 3 and img.shape[0] not in (1, 2, 3, 4):
                img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img)
            if np.issubdtype(img.dtype, np.integer):
                tensor = tensor.float() / 255.0
            else:
                tensor = tensor.float()
            return tensor

        if torch.is_tensor(img):
            tensor = img
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 3 and tensor.shape[0] not in (1, 2, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            else:
                tensor = tensor.float()
            return tensor

        raise TypeError(f"Unsupported image type: {type(img)}")


class Clamp:
    """Clamp tensor values to a finite range."""

    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, img):
        return img.clamp(self.min_value, self.max_value)


class IntensityJitter:
    """Apply lightweight brightness and contrast jitter to tensor images."""

    def __init__(self, brightness=0.1, contrast=0.1, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1).item() >= self.prob:
            return img

        brightness_scale = 1.0 + torch.empty(1).uniform_(
            -self.brightness, self.brightness
        ).item()
        contrast_scale = 1.0 + torch.empty(1).uniform_(
            -self.contrast, self.contrast
        ).item()
        mean = img.mean(dim=(-2, -1), keepdim=True)
        img = (img - mean) * contrast_scale + mean
        img = img * brightness_scale
        return img


class AdditiveGaussianNoise:
    """Inject light Gaussian noise into tensor images."""

    def __init__(self, std=0.02, prob=0.25):
        self.std = std
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1).item() >= self.prob:
            return img
        return img + torch.randn_like(img) * self.std


def get_train_transforms(
    image_size=32,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    profile="natural",
    crop_scale=(0.2, 1.0),
):
    """Get training transforms for self-supervised learning."""
    channels = len(mean) if isinstance(mean, (list, tuple)) else 3
    mean = _ensure_tuple(mean, channels)
    std = _ensure_tuple(std, channels)

    transforms_list = [
        ToFloatTensor(),
        transforms.RandomResizedCrop(image_size, scale=crop_scale, antialias=True),
    ]

    if profile == "natural":
        transforms_list.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomSolarize(threshold=0.5, p=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    elif profile == "medical":
        transforms_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=8,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                IntensityJitter(brightness=0.1, contrast=0.1, prob=0.5),
                AdditiveGaussianNoise(std=0.02, prob=0.25),
                Clamp(0.0, 1.0),
            ]
        )
    else:
        raise ValueError(f"Unknown transform profile: {profile}")

    transforms_list.append(transforms.Normalize(mean, std))
    return transforms.Compose(transforms_list)


def get_val_transforms(
    image_size=32,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
):
    """Get validation transforms."""
    channels = len(mean) if isinstance(mean, (list, tuple)) else 3
    mean = _ensure_tuple(mean, channels)
    std = _ensure_tuple(std, channels)
    return transforms.Compose(
        [
            ToFloatTensor(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize(mean, std),
        ]
    )


class ManifestImageDataset(torch.utils.data.Dataset):
    """Load image patches from a CSV manifest."""

    def __init__(self, manifest_path, split=None, transform=None):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.transform = transform
        self.samples = []

        with self.manifest_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_split = row.get("split", "").strip().lower()
                if split is not None and row_split != split.lower():
                    continue

                label_value = row.get("label", row.get("ClinSig", "")).strip()
                if label_value == "":
                    continue

                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = (self.root / image_path).resolve()

                self.samples.append(
                    {
                        "image_path": image_path,
                        "label": int(float(label_value)),
                        "patient_id": row.get("patient_id", row.get("ProxID", "")),
                        "finding_id": row.get("finding_id", row.get("fid", "")),
                    }
                )

        if not self.samples:
            split_name = split if split is not None else "all"
            raise ValueError(
                f"No samples found in manifest {self.manifest_path} for split={split_name}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path)
        return Image.open(path)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._load_image(sample["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        return image, sample["label"]


class ImageDataset(torch.utils.data.Dataset):
    """Apply augmentations multiple times to create independent views."""

    def __init__(self, dataset, transform, num_crops=2):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        views = [self.transform(image) for _ in range(self.num_crops)]
        return views, label
