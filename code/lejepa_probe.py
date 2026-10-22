"""Frozen linear evaluation and full fine-tuning on binary MiraBest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment_tracking import log_epoch, tracked_run
from checkpoint_compat import find_saved_evaluation, mirabest_subset_name

from lejepa_baseline import (
    ModelConfig,
    ViTEncoder,
    choose_device,
    load_config,
    seed_everything,
)
from mirabest_probe_data import (
    MIRABEST_CLASSES,
    PROBE_IMAGE_SHAPE,
    MiraBestProbeDataset,
    MiraBestRecord,
    build_probe_dataloader,
    build_probe_transform,
    load_mirabest_probe_records,
)


PROBE_REPRESENTATIONS = ("backbone", "projector")


def get_probe_representation(config: dict, override: str | None = None) -> str:
    representation = override if override is not None else config.get("model", {}).get("representation", "backbone")
    if representation not in PROBE_REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation!r}; choose backbone or projector")
    return representation


class LinearProbe(nn.Module):
    def __init__(self, encoder: ViTEncoder, num_classes: int, *, method="linear",
                 representation="backbone") -> None:
        super().__init__()
        if method not in ("linear", "finetune"):
            raise ValueError(f"Unknown evaluation method: {method}")
        self.method = method
        self.representation = get_probe_representation({}, representation)
        self.encoder = encoder
        feature_dim = (encoder.embed_dim if representation == "backbone" else
                       next(layer.out_features for layer in reversed(list(encoder.proj.modules()))
                            if isinstance(layer, nn.Linear)))
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes),
        )
        self.encoder.requires_grad_(False) # freezing parameters by default
        if method == "finetune":
            self.encoder.backbone.requires_grad_(True)
            if representation == "projector":
                self.encoder.proj.requires_grad_(True)
        else:
            self.encoder.eval()
        if representation == "backbone":
            self.encoder.proj.eval()

    def train(self, mode: bool = True) -> "LinearProbe":
        super().train(mode)
        if self.method == "linear":
            self.encoder.eval()
        elif self.representation == "backbone":
            self.encoder.proj.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.method == "linear":
            with torch.no_grad():
                embeddings = self._project(images)
        else:
            embeddings = self._project(images)
        return self.classifier(embeddings)

    def _project(self, images: torch.Tensor) -> torch.Tensor:
        if self.representation == "backbone":
            return self.encoder.backbone(images)
        # Keep all labelled samples: BatchNorm uses running statistics for a
        # singleton final batch, while its affine parameters still get gradients.
        singleton_norms = [layer for layer in self.encoder.proj.modules()
                           if isinstance(layer, nn.BatchNorm1d) and layer.training
                           and images.shape[0] == 1]
        for layer in singleton_norms:
            layer.eval()
        try:
            return self.encoder.proj(self.encoder.backbone(images))
        finally:
            for layer in singleton_norms:
                layer.train()


def classification_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    labels = labels.to(torch.long).cpu()
    predictions = predictions.to(torch.long).cpu()
    indices = labels * num_classes + predictions
    confusion = torch.bincount(
        indices, minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    matrix = confusion.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "accuracy": float(true_positive.sum() / matrix.sum().clamp_min(1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


class ClassificationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.labels: list[torch.Tensor] = []
        self.predictions: list[torch.Tensor] = []
        self.loss_total = 0.0
        self.count = 0

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.labels.append(labels.detach().cpu())
        self.predictions.append(logits.detach().argmax(dim=1).cpu())
        batch_size = labels.numel()
        self.loss_total += float(loss.detach()) * batch_size
        self.count += batch_size

    def compute(self) -> dict[str, float]:
        if not self.labels:
            raise RuntimeError("Probe epoch produced no batches")
        metrics = classification_metrics(
            torch.cat(self.labels),
            torch.cat(self.predictions),
            self.num_classes,
        )
        metrics["loss"] = self.loss_total / self.count
        return metrics


def run_probe_epoch(
    model: LinearProbe,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    meter = ClassificationMeter(model.classifier[-1].out_features)
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for step, (images, labels) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            meter.update(loss, logits, labels)
    return meter.compute()


def compute_class_weights(
    labels: Iterable[int], num_classes: int
) -> torch.Tensor:
    label_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes).to(torch.float32)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"Probe train split has no examples for classes {missing}")
    return label_tensor.numel() / (num_classes * counts)


def _load_pretraining_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load checkpoint metadata and normalize legacy field names."""
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=False,
    )
    if "model_config" not in checkpoint and "model_name" in checkpoint:
        checkpoint["model_config"] = {"name": checkpoint["model_name"]}
    if "state_dict" not in checkpoint and "model_state_dict" in checkpoint:
        checkpoint["state_dict"] = checkpoint["model_state_dict"]
    required = {"model_config", "augmentation", "state_dict", "data"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Pretraining checkpoint is missing fields: {sorted(missing)}")
    return checkpoint


def _load_encoder(checkpoint: dict[str, Any], representation: str = "backbone") -> ViTEncoder:
    """Construct an encoder from a normalized pretraining checkpoint."""
    encoder = ViTEncoder(ModelConfig(**checkpoint["model_config"]))
    # Older distributed runs may prefix every state key with module.
    state = {key.removeprefix("module."): value
             for key, value in checkpoint["state_dict"].items()}
    representation = get_probe_representation({}, representation)
    if representation == "backbone":
        backbone_state = {key.removeprefix("backbone."): value
                          for key, value in state.items() if key.startswith("backbone.")}
        encoder.backbone.load_state_dict(backbone_state, strict=True)
    else:
        if not any(key.startswith("proj.") for key in state):
            raise ValueError("Projector-output evaluation requires pretrained projector weights; this checkpoint contains no proj.* weights.")
        encoder.load_state_dict(state, strict=True)
    return encoder


def _split_probe_records(
    dataset_config: dict[str, Any]
) -> tuple[list[MiraBestRecord], list[MiraBestRecord], list[MiraBestRecord]]:
    return load_mirabest_probe_records(
        root=dataset_config["root"],
        validation_fraction=float(dataset_config["validation_fraction"]),
        split_seed=int(dataset_config["split_seed"]),
        confident_only=bool(dataset_config.get("confident_only", True)),
    )


def select_train_fraction(
    records: list[MiraBestRecord], fraction: float, seed: int
) -> list[MiraBestRecord]:
    selected = []
    for label in range(len(MIRABEST_CLASSES)):
        class_records = sorted(
            (record for record in records if record.label == label),
            key=lambda record: record.sample_id,
        )
        order = np.random.default_rng(seed + label).permutation(len(class_records))
        count = max(1, round(fraction * len(class_records)))
        selected.extend(class_records[int(index)] for index in order[:count])
    return selected


@dataclass
class ProbeTrainingResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_balanced_accuracy: float
    best_backbone_state: dict[str, torch.Tensor] | None
    class_weights: torch.Tensor
    test_metrics: dict[str, float]


def fit_probe(
    model: LinearProbe,
    loaders: dict[str, DataLoader],
    training_config: dict[str, Any],
    train_label_indices: list[int],
    device: torch.device,
    *,
    max_steps_override: int | None = None,
    max_eval_steps_override: int | None = None,
    run=None,
) -> ProbeTrainingResult:
    """Train, select by validation, restore best weights, then evaluate test data."""
    parameters = [{"params": model.classifier.parameters(),
                   "lr": float(training_config["learning_rate"])}]

    if model.method == "finetune":
        encoder_parameters = (model.encoder.backbone.parameters() if model.representation == "backbone"
                              else model.encoder.parameters())
        parameters.append({"params": encoder_parameters,
                           "lr": float(training_config.get("backbone_learning_rate", 1e-5))})
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    class_weighting = bool(training_config.get("class_weighting", True))
    class_weights = (
        compute_class_weights(train_label_indices, model.classifier[-1].out_features)
        if class_weighting
        else torch.ones(model.classifier[-1].out_features, dtype=torch.float32)
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    epochs = int(training_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training_config.get("minimum_learning_rate", 0.0)),
    )
    patience = int(training_config.get("patience", epochs))
    history: list[dict[str, Any]] = []
    best_epoch = 0
    epochs_without_improvement = 0
    best_validation_balanced_accuracy = -1.0
    best_classifier_state: dict[str, torch.Tensor] | None = None
    best_backbone_state = None
    best_projector_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = run_probe_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_steps=max_steps_override,
        )
        validation_metrics = run_probe_epoch(
            model,
            loaders["validation"],
            criterion,
            device,
            max_steps=max_eval_steps_override,
        )
        metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(metrics)
        log_epoch(run, metrics)
        score = validation_metrics["balanced_accuracy"]
        if score > best_validation_balanced_accuracy:
            best_validation_balanced_accuracy = score
            best_epoch = epoch
            best_classifier_state = deepcopy(model.classifier.state_dict())
            if model.method == "finetune":
                best_backbone_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.encoder.backbone.state_dict().items()
                }
                if model.representation == "projector":
                    best_projector_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.encoder.proj.state_dict().items()
                    }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            break

    if best_classifier_state is None:
        raise RuntimeError("The MiraBest probe did not complete an epoch")
    model.classifier.load_state_dict(best_classifier_state)
    if model.method == "finetune":
        model.encoder.backbone.load_state_dict(best_backbone_state)
        if model.representation == "projector":
            model.encoder.proj.load_state_dict(best_projector_state)
    test_metrics = run_probe_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        max_steps=max_eval_steps_override,
    )
    return ProbeTrainingResult(
        history=history, best_epoch=best_epoch,
        best_validation_balanced_accuracy=best_validation_balanced_accuracy,
        best_backbone_state=best_backbone_state, class_weights=class_weights,
        test_metrics=test_metrics,
    )


def train_probe(
    config: dict[str, Any],
    *,
    checkpoint_override: str | Path | None = None,
    max_steps_override: int | None = None,
    max_eval_steps_override: int | None = None,
    output_dir_override: str | Path | None = None,
    train_fraction_override: float | None = None,
    method: str = "linear",
    representation_override: str | None = None,
) -> Path:
    """Prepare data/model, delegate training to fit_probe, and save results."""
    dataset_config = config["dataset"]
    confident_only = bool(dataset_config.get("confident_only", True))
    subset = mirabest_subset_name(confident_only)
    representation = get_probe_representation(config, representation_override)
    if method not in ("linear", "finetune"):
        raise ValueError(f"Unknown evaluation method: {method}")
    training_config = dict(config["training"])
    if method == "finetune":
        training_config.update(config.get("finetuning", {}))
    checkpoint_path = checkpoint_override or config["model"]["checkpoint"]
    train_fraction = float(1.0 if train_fraction_override is None else train_fraction_override)
    if not 0 < train_fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1].")
    checkpoint_path = Path(checkpoint_path).expanduser().parent / "checkpoint_best_validation.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MiraBest training requires the best-validation encoder: {checkpoint_path}")
    output_dir = Path(
        output_dir_override or config["output"]["directory"]
    ).expanduser()
    if output_dir_override is None and method == "finetune":
        output_dir = output_dir.parent / "finetune"
    checkpoint = _load_pretraining_checkpoint(checkpoint_path)
    model_name = checkpoint.get("model_config", {}).get(
        "name", checkpoint.get("model_name"))
    overrides = dict(config["training"].get("model_learning_rates", {}).get(model_name, {}))
    if method == "finetune":
        overrides.update(config.get("finetuning", {}).get("model_learning_rates", {}).get(model_name, {}))
    unexpected = set(overrides) - {"learning_rate", "backbone_learning_rate"}
    if unexpected:
        raise ValueError(f"Unknown model learning-rate overrides: {sorted(unexpected)}")
    training_config.update(overrides)
    pretraining_epoch = checkpoint.get("epoch")
    stem = "linear_probe" if method == "linear" else "finetune"
    if bool(config.get("scaling", {}).get("skip_completed_runs", True)):
        existing = find_saved_evaluation(
            output_dir, f"{stem}_metrics.json", representation,
            confident_only=confident_only, pretraining_epoch=pretraining_epoch,
            train_fraction=train_fraction, require_checkpoint=True,
        )
        if existing is not None:
            return existing.with_name(f"{stem}.pt")
    if pretraining_epoch is not None:
        output_dir = output_dir / f"epochs_{pretraining_epoch}"
    output_dir = output_dir / representation / subset
    output_path = output_dir / f"{stem}.pt"
    metrics_path = output_dir / f"{stem}_metrics.json"

    seed = int(training_config["seed"])
    seed_everything(seed)
    device = choose_device(str(training_config.get("device", "auto")))
    encoder = _load_encoder(checkpoint, representation)
    transform = build_probe_transform(checkpoint["augmentation"])

    full_train_records, validation_records, test_records = _split_probe_records(
        dataset_config
    )
    train_records = select_train_fraction(full_train_records, train_fraction, seed)
    labels = list(MIRABEST_CLASSES)
    datasets = {
        "train": MiraBestProbeDataset(train_records, transform),
        "validation": MiraBestProbeDataset(validation_records, transform),
        "test": MiraBestProbeDataset(test_records, transform),
    }
    batch_size = int(training_config["batch_size"])
    num_workers = int(dataset_config.get("num_workers", 0))
    loaders = {
        role: build_probe_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=role == "train",
            num_workers=num_workers,
            seed=seed,
            device=device,
        )
        for role, dataset in datasets.items()
    }

    model = LinearProbe(encoder, len(labels), method=method,
                        representation=representation).to(device)
    with tracked_run(
        config.get("wandb", {}), output_dir,
        name=f"{method}-{representation}-{subset}-{model_name}-rgz_train_{checkpoint['data'].get('train_size', checkpoint['data']['sample_size'])}-val_{checkpoint['data'].get('validation_size', 0)}-mb_train_fraction_{train_fraction:g}-seed_{seed}-pretrain_epoch_{pretraining_epoch}",
        job_type=method,
        config={"stage": method, "representation": representation,
                "architecture": model_name, "dataset": subset,
                **training_config, "train_fraction": train_fraction,
                "pretraining_epoch": pretraining_epoch,
                "pretraining_data": checkpoint["data"],
                "encoder_checkpoint": str(checkpoint_path),
                "split_sizes": {role: len(dataset) for role, dataset in datasets.items()},
                "max_train_steps": max_steps_override, "max_eval_steps": max_eval_steps_override},
    ) as run:
        result = fit_probe(
            model, loaders, training_config, [record.label for record in train_records], device,
            max_steps_override=max_steps_override,
            max_eval_steps_override=max_eval_steps_override, run=run,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_metadata = {
            "source": "MiraBest binary FR-I/FR-II benchmark",
            "subset": subset,
            "confident_only": confident_only,
            "root": str(dataset_config["root"]),
            "preprocessing": "published sigma-clipped uint8 images scaled to [0, 1]",
            "image_shape": list(PROBE_IMAGE_SHAPE),
            "train_batches": [f"data_batch_{index}" for index in range(1, 8)],
            "test_batch": "test_batch",
            "excluded_fine_labels": [3, 4, 7, 8, 9] if confident_only else [8, 9],
            "train_fraction": train_fraction,
            "available_train_samples": len(full_train_records),
            "split_sizes": {
                "train": len(train_records),
                "validation": len(validation_records),
                "test": len(test_records),
            },
            "class_names": labels,
        }
        torch.save(
            {
                "method": method,
                "representation": representation,
                "training_config": training_config,
                "backbone_state_dict": result.best_backbone_state,
                "projector_state_dict": ({key: value.detach().cpu() for key, value in model.encoder.proj.state_dict().items()}
                                         if representation == "projector" else None),
                "encoder_checkpoint": str(checkpoint_path),
                "encoder_model": checkpoint["model_config"],
                "pretraining_data": checkpoint["data"],
                "pretraining_epoch": pretraining_epoch,
                "probe_data": probe_metadata,
                "class_weights": result.class_weights.tolist(),
                "classifier_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.classifier.state_dict().items()
                },
                "best_epoch": result.best_epoch,
                "history": result.history,
                "test": result.test_metrics,
            },
            output_path,
        )
        metrics_path.write_text(
            json.dumps(
                {
                    "method": method,
                    "representation": representation,
                    "training_config": training_config,
                    "encoder_checkpoint": str(checkpoint_path),
                    "pretraining_data": checkpoint["data"],
                    "pretraining_epoch": pretraining_epoch,
                    "probe_data": probe_metadata,
                    "best_epoch": result.best_epoch,
                    "class_weights": result.class_weights.tolist(),
                    "history": result.history,
                    "test": result.test_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if run is not None:
            run.log({f"test/{key}": value for key, value in result.test_metrics.items()})
            run.summary.update({"best_epoch": result.best_epoch,
                                "best_validation_balanced_accuracy": result.best_validation_balanced_accuracy,
                                "completed_epochs": len(result.history),
                                "early_stopped": len(result.history) < int(training_config["epochs"]),
                                "checkpoint": str(output_path)})
        return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("probe_config.yaml")),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--method", choices=("linear", "finetune"), default="linear")
    parser.add_argument("--representation", choices=PROBE_REPRESENTATIONS,
                        help="Override model.representation (default: backbone).")
    parser.add_argument("--confident-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Use certain sources; --no-confident-only restores the previous selection.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    config = load_config(arguments.config)
    if arguments.confident_only is not None:
        config["dataset"]["confident_only"] = arguments.confident_only
    output = train_probe(
        config,
        checkpoint_override=arguments.checkpoint,
        max_steps_override=arguments.max_steps,
        max_eval_steps_override=arguments.max_eval_steps,
        output_dir_override=arguments.output_dir,
        train_fraction_override=arguments.train_fraction,
        method=arguments.method,
        representation_override=arguments.representation,
    )
    print(f"Saved MiraBest {arguments.method} evaluation: {output}")
