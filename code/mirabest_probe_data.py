"""Binary MiraBest data with confident-only or legacy all-source selection."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


MIRABEST_CLASSES = ("FR-I", "FR-II")
PROBE_IMAGE_SHAPE = (150, 150)
TRAIN_BATCHES = tuple(f"data_batch_{index}" for index in range(1, 8))
TEST_BATCH = "test_batch"


@dataclass(frozen=True)
class MiraBestRecord:
    sample_id: str
    path: Path
    batch_index: int
    split: str
    label: int


def _binary_label(label: int, *, confident_only: bool = True) -> int | None:
    """Map certain sources by default, or include uncertain sources as before."""
    if label in ((0, 1, 2) if confident_only else (0, 1, 2, 3, 4)):
        return 0
    if label in ((5, 6) if confident_only else (5, 6, 7)):
        return 1
    return None


@lru_cache(maxsize=8)
def _load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    images = np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 150, 150)
    labels = np.asarray(batch["labels"], dtype=np.int64)
    return images, labels


def _load_records(
    root: Path, batch_names: Sequence[str], split: str, *, confident_only: bool = True,
) -> list[MiraBestRecord]:
    records = []
    for batch_name in batch_names:
        path = root / batch_name
        _, labels = _load_batch(str(path.resolve()))
        for index, fine_label in enumerate(labels):
            label = _binary_label(int(fine_label), confident_only=confident_only)
            if label is not None:
                records.append(
                    MiraBestRecord(
                        sample_id=f"{batch_name}/{index:03d}",
                        path=path,
                        batch_index=index,
                        split=split,
                        label=label,
                    )
                )
    return records


def load_mirabest_probe_records(
    root: str | Path,
    *,
    validation_fraction: float,
    split_seed: int,
    confident_only: bool = True,
) -> tuple[list[MiraBestRecord], list[MiraBestRecord], list[MiraBestRecord]]:
    root = Path(root).expanduser().resolve()
    published_train = _load_records(root, TRAIN_BATCHES, "train", confident_only=confident_only)
    test = _load_records(root, (TEST_BATCH,), "test", confident_only=confident_only)

    order = np.random.default_rng(split_seed).permutation(len(published_train))
    validation_size = round(validation_fraction * len(published_train))
    validation_indices = set(int(index) for index in order[:validation_size])
    train = [
        record
        for index, record in enumerate(published_train)
        if index not in validation_indices
    ]
    validation = [
        replace(record, split="validation")
        for index, record in enumerate(published_train)
        if index in validation_indices
    ]
    return train, validation, test


def load_mirabest_image(record: MiraBestRecord) -> torch.Tensor:
    image = _load_batch(str(record.path.resolve()))[0][record.batch_index]
    image = torch.from_numpy(image.copy()).float().div(255.0)
    return image.unsqueeze(0).repeat(3, 1, 1)


def build_probe_transform(values: dict[str, Any]) -> v2.Compose:
    image_size = int(values["image_size"])
    return v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.Normalize(
                tuple(values["normalization_mean"]),
                tuple(values["normalization_std"]),
            ),
        ]
    )


class MiraBestProbeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[MiraBestRecord],
        transform: v2.Compose,
    ) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        return self.transform(load_mirabest_image(record)), record.label


def build_probe_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
