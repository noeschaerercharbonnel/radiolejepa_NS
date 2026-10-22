"""Visualize the RGZ20k pre-training and MiraBest probe datasets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lejepa_baseline import load_config
from rgz20k_data import (
    RGZLeJEPADataset,
    augmentation_from_config,
    load_rgz20k_records,
)
from mirabest_probe_data import (
    MIRABEST_CLASSES,
    MiraBestRecord,
    load_mirabest_image,
    load_mirabest_probe_records,
)


def denormalize(
    view: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    mean_tensor = torch.tensor(mean, dtype=view.dtype)[:, None, None]
    std_tensor = torch.tensor(std, dtype=view.dtype)[:, None, None]
    image = (view.cpu() * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def save_augmentation_grid(
    dataset: RGZLeJEPADataset,
    output_path: str | Path,
    *,
    index: int = 0,
) -> Path:
    views, sample_id = dataset[index]
    number_of_images = 1 + views.size(0)
    columns = 4
    rows = math.ceil(number_of_images / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    axes[0].imshow(dataset.get_image(index), cmap="gray")
    axes[0].set_title("Original radio cutout")
    axes[0].axis("off")

    global_count = dataset.augmentation.global_views
    for view_index, view in enumerate(views):
        axis = axes[view_index + 1]
        axis.imshow(
            denormalize(
                view,
                dataset.augmentation.normalization_mean,
                dataset.augmentation.normalization_std,
            )
        )
        title = (
            f"Global {view_index + 1}"
            if view_index < global_count
            else f"Local {view_index - global_count + 1}"
        )
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[number_of_images:]:
        axis.axis("off")
    figure.suptitle(f"RGZ LeJEPA views — {sample_id}", fontsize=14)
    figure.tight_layout()
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def save_probe_grid(
    records: list[MiraBestRecord],
    output_path: str | Path,
    *,
    examples_per_class: int = 2,
) -> Path:
    selected = [
        record
        for label in range(len(MIRABEST_CLASSES))
        for record in [item for item in records if item.label == label][
            :examples_per_class
        ]
    ]
    columns = examples_per_class
    rows = len(MIRABEST_CLASSES)
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for axis, record in zip(axes.reshape(-1), selected):
        axis.imshow(
            load_mirabest_image(record)[0], cmap="gray", vmin=0.0, vmax=1.0
        )
        axis.set_title(f"{MIRABEST_CLASSES[record.label]} — {record.sample_id}")
        axis.axis("off")
    for axis in axes.reshape(-1)[len(selected) :]:
        axis.axis("off")
    figure.suptitle("MiraBest binary FR-I/FR-II probe", fontsize=14)
    figure.tight_layout()
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def load_probe_records(config: dict) -> list[MiraBestRecord]:
    values = config["dataset"]
    train, validation, test = load_mirabest_probe_records(
        root=values["root"],
        validation_fraction=float(values["validation_fraction"]),
        split_seed=int(values["split_seed"]),
        confident_only=bool(values.get("confident_only", True)),
    )
    return train + validation + test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pretraining-config",
        default=str(Path(__file__).with_name("baseline_config.yaml")),
    )
    parser.add_argument(
        "--probe-config",
        default=str(Path(__file__).with_name("probe_config.yaml")),
    )
    parser.add_argument(
        "--dataset",
        choices=("pretraining", "probe", "both"),
        default="both",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--pretraining-output",
        default="../outputs/rgz20k_augmentation_grid.png",
    )
    parser.add_argument(
        "--probe-output",
        default="../outputs/mirabest_probe_grid.png",
    )
    parser.add_argument("--examples-per-class", type=int, default=2)
    parser.add_argument("--confident-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Use certain sources; --no-confident-only restores the previous selection.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.dataset in ("pretraining", "both"):
        config = load_config(arguments.pretraining_config)
        records = load_rgz20k_records(config["dataset"]["batches_root"])
        dataset = RGZLeJEPADataset(
            records,
            augmentation_from_config(config["views"]),
        )
        saved = save_augmentation_grid(
            dataset,
            arguments.pretraining_output,
            index=arguments.index,
        )
        print(f"Saved RGZ20k augmentation grid: {saved}")
    if arguments.dataset in ("probe", "both"):
        config = load_config(arguments.probe_config)
        if arguments.confident_only is not None:
            config["dataset"]["confident_only"] = arguments.confident_only
        saved = save_probe_grid(
            load_probe_records(config),
            arguments.probe_output,
            examples_per_class=arguments.examples_per_class,
        )
        print(f"Saved MiraBest probe grid: {saved}")
