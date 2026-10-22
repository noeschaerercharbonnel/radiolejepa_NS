"""RGZ20k loading and LeJEPA view generation for pre-training."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


BATCH_NAMES = tuple(f"data_batch_{index}" for index in range(1, 11))


@dataclass(frozen=True)
class RGZRecord:
    sample_id: str
    path: Path
    batch_index: int


@dataclass(frozen=True)
class RadioAugmentationConfig:
    image_size: int
    local_image_size: int
    global_views: int
    local_views: int
    global_crop_scale: tuple[float, float]
    local_crop_scale: tuple[float, float]
    color_jitter_probability: float
    brightness_jitter: float
    contrast_jitter: float
    saturation_jitter: float
    hue_jitter: float
    blur_probability: float
    blur_kernel_size: int
    blur_sigma: tuple[float, float]
    horizontal_flip_probability: float
    normalization_mean: tuple[float, float, float]
    normalization_std: tuple[float, float, float]

    @property
    def total_views(self) -> int:
        return self.global_views + self.local_views

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def augmentation_from_config(values: dict[str, Any]) -> RadioAugmentationConfig:
    values = dict(values)
    for name in (
        "global_crop_scale",
        "local_crop_scale",
        "blur_sigma",
        "normalization_mean",
        "normalization_std",
    ):
        values[name] = tuple(values[name])
    return RadioAugmentationConfig(**values)


def load_rgz20k_records(root: str | Path) -> list[RGZRecord]:
    root = Path(root).expanduser().resolve()
    return [
        RGZRecord(f"{batch_name}/{index:04d}", root / batch_name, index)
        for batch_name in BATCH_NAMES
        for index in range(2_000)
    ]


def deterministic_record_order(
    records: Sequence[RGZRecord], seed: int
) -> list[RGZRecord]:
    records = sorted(records, key=lambda record: record.sample_id)
    order = np.random.default_rng(seed).permutation(len(records))
    return [records[int(index)] for index in order]


def select_nested_subset(
    records: Sequence[RGZRecord], sample_size: int, seed: int
) -> list[RGZRecord]:
    return deterministic_record_order(records, seed)[:sample_size]


def split_pretraining_records(records, validation_fraction=0.20):
    """Split the seeded subset order, preserving nested train/validation sets."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    train, validation = [], []
    for index, record in enumerate(records):
        held_out = int((index + 1) * validation_fraction) > int(index * validation_fraction)
        (validation if held_out else train).append(record)
    if validation_fraction > 0 and not validation:
        raise ValueError("The subset is too small for the validation fraction.")
    return train, validation


@lru_cache(maxsize=2)
def _load_batch(path: str) -> np.ndarray:
    with Path(path).open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")["data"]
    return np.asarray(data, dtype=np.uint8).reshape(2_000, 150, 150)

# encoding="latin1" is used because these RGZ batches were likely pickled with Python 2. It lets Python 3 decode the old byte-string keys and values without Unicode errors.



def load_rgz20k_image(record: RGZRecord) -> Image.Image:
    image = _load_batch(str(record.path.resolve()))[record.batch_index]
    return Image.fromarray(image).convert("RGB")


def _color_jitter(config: RadioAugmentationConfig) -> v2.RandomApply:
    return v2.RandomApply(
        [
            v2.ColorJitter(
                config.brightness_jitter,
                config.contrast_jitter,
                config.saturation_jitter,
                config.hue_jitter,
            )
        ],
        p=config.color_jitter_probability,
    )


def _blur(config: RadioAugmentationConfig) -> v2.RandomApply:
    return v2.RandomApply(
        [v2.GaussianBlur(config.blur_kernel_size, sigma=config.blur_sigma)],
        p=config.blur_probability,
    )


def _normalization(config: RadioAugmentationConfig) -> list[Any]:
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(config.normalization_mean, config.normalization_std),
    ]


def build_global_transform(config: RadioAugmentationConfig) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                config.image_size,
                scale=config.global_crop_scale,
                antialias=True,
            ),
            _color_jitter(config),
            _blur(config),
            v2.RandomHorizontalFlip(config.horizontal_flip_probability),
            *_normalization(config),
        ]
    )


def build_local_transform(config: RadioAugmentationConfig) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomResizedCrop(
                config.local_image_size,
                scale=config.local_crop_scale,
                antialias=True,
            ),
            _color_jitter(config),
            _blur(config),
            v2.RandomHorizontalFlip(config.horizontal_flip_probability),
            v2.Resize((config.image_size, config.image_size), antialias=True),
            *_normalization(config),
        ]
    )


class RGZLeJEPADataset(Dataset):
    def __init__(
        self,
        records: Sequence[RGZRecord],
        augmentation: RadioAugmentationConfig,
    ) -> None:
        self.records = list(records)
        self.augmentation = augmentation
        self.global_transform = build_global_transform(augmentation)
        self.local_transform = build_local_transform(augmentation)

    def __len__(self) -> int:
        return len(self.records)

    def get_image(self, index: int) -> Image.Image:
        return load_rgz20k_image(self.records[index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image = self.get_image(index)
        views = [
            self.global_transform(image)
            for _ in range(self.augmentation.global_views)
        ]
        views += [
            self.local_transform(image)
            for _ in range(self.augmentation.local_views)
        ]
        return torch.stack(views), self.records[index].sample_id


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
    drop_last: bool,
    persistent_workers: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers and num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
