"""LeJEPA pretraining on deterministic nested Radio Galaxy Zoo subsets."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.ops import MLP

from experiment_tracking import log_epoch, tracked_run
from checkpoint_compat import pretraining_data_directory

from rgz20k_data import (
    RGZLeJEPADataset,
    RGZRecord,
    augmentation_from_config,
    build_dataloader,
    deterministic_record_order,
    load_rgz20k_records,
    select_nested_subset,
    split_pretraining_records,
)


SUPPORTED_MODELS = (
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "vit_tiny_patch16_224"
    image_size: int = 224
    projection_dim: int = 128
    mlp_hidden_dim: int = 2048
    drop_path_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model {self.name!r}; choose one of {SUPPORTED_MODELS}"
            )


class SIGReg(nn.Module):
    """Erica-compatible sliced Epps-Pulley regularizer."""

    def __init__(self, knots: int = 17, num_directions: int = 256) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_directions = num_directions
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        directions = torch.randn(
            projections.size(-1),
            self.num_directions,
            device=projections.device,
            dtype=projections.dtype,
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
        x_t = (projections @ directions).unsqueeze(-1) * self.t
        error = (
            (x_t.cos().mean(-3) - self.phi).square()
            + x_t.sin().mean(-3).square()
        )
        return ((error @ self.weights) * projections.size(-2)).mean()


class ViTEncoder(nn.Module):
    """A native timm ViT backbone plus the retained LeJEPA projector."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        from timm import create_model

        self.config = config or ModelConfig()
        self.backbone = create_model(
            self.config.name,
            pretrained=False,
            num_classes=0,
            drop_path_rate=self.config.drop_path_rate,
            img_size=self.config.image_size,
        )
        self.embed_dim = int(self.backbone.num_features)
        self.proj = MLP(
            self.embed_dim,
            [
                self.config.mlp_hidden_dim,
                self.config.mlp_hidden_dim,
                self.config.projection_dim,
            ],
            norm_layer=nn.BatchNorm1d,
        )

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = views.shape[:2]
        embeddings = self.backbone(views.flatten(0, 1))
        projections = self.proj(embeddings)
        projections = projections.reshape(batch_size, num_views, -1).transpose(0, 1)
        return embeddings, projections


def lejepa_loss(
    projections: torch.Tensor,
    sigreg: SIGReg,
    lambda_sigreg: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep the original all-view center and weighted LeJEPA objective fixed."""

    invariance = (projections.mean(0) - projections).square().mean()
    regularization = sigreg(projections)
    total = lambda_sigreg * regularization + (1.0 - lambda_sigreg) * invariance
    projection_std = projections.flatten(0, 1).float().std(
        dim=0, unbiased=False
    ).mean()
    return total, {
        "loss": total,
        "sigreg": regularization,
        "invariance": invariance,
        "projection_std": projection_std,
    }


class MetricAverager:
    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        for name, value in metrics.items():
            self.totals[name] = (
                self.totals.get(name, 0.0) + float(value.detach()) * batch_size
            )
        self.count += batch_size

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise RuntimeError("LeJEPA epoch produced no batches")
        return {name: total / self.count for name, total in self.totals.items()}


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_epoch(
    model: ViTEncoder,
    sigreg: SIGReg,
    loader: DataLoader,
    device: torch.device,
    lambda_sigreg: float,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    meter = MetricAverager()
    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for step, (views, _) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            views = views.to(device, non_blocking=device.type == "cuda")
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device):
                _, projections = model(views)
                loss, metrics = lejepa_loss(projections, sigreg, lambda_sigreg)
            if is_training:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            meter.update(metrics, views.size(0))
    return meter.compute()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config


def run_validation_epoch(model, sigreg, loader, device, lambda_sigreg, *, seed, max_steps):
    """Repeat validation views/directions without changing training's RNG stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generator_state = loader.generator.get_state()
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            loader.generator.manual_seed(seed)
            return run_epoch(model, sigreg, loader, device, lambda_sigreg,
                             max_steps=max_steps)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        loader.generator.set_state(generator_state)


def _load_pretraining_records(dataset_config: dict[str, Any]) -> list[RGZRecord]:
    records = load_rgz20k_records(dataset_config["batches_root"])
    if len(records) != 20_000:
        raise ValueError(
            "The official RGZ20k training batches must contain exactly "
            f"20,000 records, found {len(records):,}"
        )
    return records


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    minimum_learning_rate: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if warmup_steps <= 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=minimum_learning_rate,
        )
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=minimum_learning_rate,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _save_checkpoint(
    path: Path,
    *,
    model: ViTEncoder,
    model_config: ModelConfig,
    augmentation: dict[str, Any],
    history: list[dict[str, Any]],
    epoch: int,
    data_metadata: dict[str, Any],
    optimizer=None,
    scheduler=None,
    training_config=None,
    steps_per_epoch=None,
    loader=None,
    sample_ids=None,
    best_validation_loss=None,
    best_validation_epoch=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model_name": model_config.name,
            "model_config": asdict(model_config),
            "embedding_dim": model.embed_dim,
            "augmentation": augmentation,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": history,
            "data": data_metadata,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "training_config": training_config,
            "steps_per_epoch": steps_per_epoch,
            "sample_ids": sample_ids,
            "best_validation_loss": best_validation_loss,
            "best_validation_epoch": best_validation_epoch,
            "rng": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader": loader.generator.get_state() if loader is not None and loader.generator is not None else None,
            },
        },
        path.with_suffix(".pt.tmp"),
    )
    path.with_suffix(".pt.tmp").replace(path)
    if path.name == "checkpoint_last.pt":
        metrics_path = path.parent / "metrics.json"
        temporary = metrics_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "format_version": 2, "model": asdict(model_config),
            "parameter_count": sum(p.numel() for p in model.backbone.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
            "data": data_metadata, "history": history, "epoch": epoch,
            "target_epochs": training_config["epochs"], "checkpoint": str(path),
            "best_validation_loss": best_validation_loss,
            "best_validation_epoch": best_validation_epoch,
        }, indent=2), encoding="utf-8")
        temporary.replace(metrics_path)


def resume_pretraining(
    resume_path: Path,
    *,
    model: ViTEncoder,
    model_config: ModelConfig,
    augmentation,
    optimizer,
    scheduler,
    training_config: dict[str, Any],
    steps_per_epoch: int,
    train_loader: DataLoader,
    data_metadata: dict[str, Any],
    sample_ids: list[str],
    train_records: list[RGZRecord],
    output_dir: Path,
) -> tuple[int, list[dict[str, Any]]]:
    """Restore compatible epoch-boundary state and write the subset manifest.

    A missing checkpoint starts at epoch zero. A completed checkpoint returns
    its epoch/history so the caller can skip training and W&B initialization.
    """
    epochs = int(training_config["epochs"])
    history = []
    start_epoch = 0
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("data", {}).get("validation_sample_ids", []) != data_metadata["validation_sample_ids"]:
            raise ValueError("Checkpoint uses a different pretraining validation split. Use a new output directory for a fresh run; old training data cannot become held-out validation data.")
        start_epoch = int(saved["epoch"])
        if start_epoch >= epochs:
            print(f"Already trained through epoch {start_epoch}: {resume_path}")
            return start_epoch, list(saved.get("metrics", []))
        if not saved.get("optimizer") or not saved.get("scheduler") or not saved.get("rng"):
            raise ValueError("This older checkpoint has no complete training state; cannot resume without resetting AdamW. Use a new output directory for a fresh run.")
        if saved["model_config"] != asdict(model_config) or saved["augmentation"] != augmentation.to_dict() or saved.get("sample_ids") != sample_ids:
            raise ValueError("Resume requires the same model, augmentation, and data subset.")
        old_training = saved["training_config"]
        # Ignore removed safety settings when reading older checkpoints.
        ignored = {"epochs", "device", "checkpoint_interval", "collapse_std_threshold", "collapse_warmup_epochs"}
        if {k: v for k, v in old_training.items() if k not in ignored} != {k: v for k, v in training_config.items() if k not in ignored} or saved["steps_per_epoch"] != steps_per_epoch:
            raise ValueError("Resume only supports changing the epoch target, device, or checkpoint settings; keep other training settings unchanged.")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if epochs != int(old_training["epochs"]):
            # Stretch the existing cosine schedule, retaining optimizer moments/steps.
            cosine = scheduler._schedulers[-1] if isinstance(scheduler, SequentialLR) else scheduler
            # cosine duration = updates per epoch × epochs after warmup
            cosine.T_max = max(steps_per_epoch * (epochs - int(training_config.get("warmup_epochs", 0))), 1)
            if not isinstance(scheduler, SequentialLR) or scheduler.last_epoch >= steps_per_epoch * int(training_config.get("warmup_epochs", 0)):
                # implements the cosine schedule
                rates = [cosine.eta_min + (base - cosine.eta_min) * (1 + math.cos(math.pi * cosine.last_epoch / cosine.T_max)) / 2 for base in cosine.base_lrs]
                for group, rate in zip(optimizer.param_groups, rates):
                    group["lr"] = rate
                cosine._last_lr = rates
                scheduler._last_lr = rates
            print(f"Extending epoch target {old_training['epochs']} -> {epochs}; AdamW state preserved, cosine schedule stretched.")
        rng = saved["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng["loader"] is not None and train_loader.generator is not None:
            train_loader.generator.set_state(rng["loader"])
        history = list(saved["metrics"])
        print(f"Resuming after epoch {start_epoch}: {resume_path}")
    (output_dir / "subset.json").write_text(
        json.dumps(
            {
                **data_metadata,
                "sample_ids": sample_ids,
                "train_sample_ids": [record.sample_id for record in train_records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return start_epoch, history


def fit_pretraining(
    model: ViTEncoder,
    sigreg: SIGReg,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    training_config: dict[str, Any],
    output_dir: Path,
    start_epoch: int,
    history: list[dict[str, Any]],
    checkpoint_options: dict[str, Any],
    max_steps: int | None = None,
    validation_seed: int = 15,
    run=None,
) -> Path:
    """Run pretraining epochs, monitor validation, and save latest/best models.

    Model construction and checkpoint restoration are handled by the caller.
    Historical validation metrics retain the best loss across resumed runs.
    """
    epochs = int(training_config["epochs"])
    last_path = output_dir / "checkpoint_last.pt"
    save_options = dict(checkpoint_options)
    lambda_sigreg = float(training_config["lambda_sigreg"])
    checkpoint_interval = int(training_config.get("checkpoint_interval", 10))
    best_path = output_dir / "checkpoint_best_validation.pt"
    previous_validation = [row for row in history
                           if math.isfinite(row.get("validation", {}).get("loss", math.inf))]
    best_row = min(previous_validation, key=lambda row: row["validation"]["loss"], default=None)
    best_validation_loss = best_row["validation"]["loss"] if best_row else math.inf
    best_validation_epoch = best_row["epoch"] if best_row else None
    for epoch in range(start_epoch + 1, epochs + 1):
        training_metrics = run_epoch(
            model,
            sigreg,
            train_loader,
            device,
            lambda_sigreg,
            optimizer=optimizer,
            scheduler=scheduler,
            max_steps=max_steps,
        )
        validation_metrics = (run_validation_epoch(
            model, sigreg, validation_loader, device, lambda_sigreg,
            seed=validation_seed, max_steps=max_steps,
        ) if validation_loader is not None else {})
        metrics = {
            "epoch": epoch,
            "train": training_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(metrics)
        log_epoch(run, metrics)

        improved = validation_metrics.get("loss", math.inf) < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_validation_epoch = epoch
        save_options.update(
            best_validation_loss=best_validation_loss if best_validation_epoch is not None else None,
            best_validation_epoch=best_validation_epoch,
        )
        if improved:
            _save_checkpoint(best_path, history=history, epoch=epoch, **save_options)
        _save_checkpoint(last_path, history=history, epoch=epoch, **save_options)
        print(f"Epoch {epoch}/{epochs}: checkpoint saved")

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            _save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                history=history,
                epoch=epoch,
                **save_options,
            )

    if run is not None:
        run.summary.update({"completed_epochs": history[-1]["epoch"],
                            "checkpoint": str(last_path),
                            "best_validation_epoch": best_validation_epoch,
                            "best_validation_loss": save_options["best_validation_loss"],
                            "best_validation_checkpoint": str(best_path) if best_validation_epoch is not None else None})

    return last_path


def train(
    config: dict[str, Any],
    *,
    sample_size_override: int | None = None,
    model_name_override: str | None = None,
    max_steps_override: int | None = None,
    output_dir_override: str | Path | None = None,
    resume_checkpoint_override: str | Path | None = None,
) -> Path:
    """Prepare data/model, restore training state, then delegate to fit_pretraining."""
    dataset_config = config["dataset"]
    training_config = config["training"]
    model_values = dict(config["model"])
    if model_name_override is not None:
        model_values["name"] = model_name_override
    model_config = ModelConfig(**model_values)
    augmentation = augmentation_from_config(config["views"])
    if augmentation.image_size != model_config.image_size:
        raise ValueError("model.image_size and views.image_size must match")

    seed = int(training_config["seed"])
    seed_everything(seed)
    device = choose_device(str(training_config.get("device", "auto")))
    train_pool = _load_pretraining_records(dataset_config)
    validation_fraction = float(dataset_config.get("validation_fraction", 0.20))
    split_seed = int(dataset_config.get("split_seed", 14))
    available_train_records, validation_records = split_pretraining_records(
        deterministic_record_order(train_pool, split_seed), validation_fraction,
    )
    configured_size = dataset_config.get("sample_size")
    sample_size = (int(sample_size_override) if sample_size_override is not None
                   else int(configured_size if configured_size is not None else len(available_train_records)))
    if not 0 < sample_size <= len(available_train_records):
        raise ValueError(
            f"sample_size is the actual training count and must be between 1 and "
            f"{len(available_train_records)} after reserving {len(validation_records)} validation images."
        )
    train_records = select_nested_subset(available_train_records, sample_size, seed)
    selected_records = train_records
    print(f"RGZ split: training={len(train_records):,}, validation={len(validation_records):,}, "
          f"full_pool={len(train_pool):,}; model={model_config.name}")
    train_dataset = RGZLeJEPADataset(train_records, augmentation)
    batch_size = int(training_config["batch_size"])
    num_workers = int(dataset_config.get("num_workers", 0))
    if batch_size < 4:
        raise ValueError(
            "SIGReg requires a meaningful object batch; batch_size must be at least 4"
        )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        device=device,
        drop_last=True,
        persistent_workers=False,  # Epoch-boundary worker seeds are resumable.
    )
    validation_loader = None
    if validation_records:
        validation_loader = build_dataloader(
            RGZLeJEPADataset(validation_records, augmentation),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            seed=split_seed + 1, device=device, drop_last=False, persistent_workers=False,
        )
    configured_max_steps = training_config.get("max_steps_per_epoch")
    max_steps = (
        max_steps_override
        if max_steps_override is not None
        else configured_max_steps
    )
    max_steps = int(max_steps) if max_steps is not None else None
    steps_per_epoch = min(len(train_loader), max_steps or len(train_loader))
    if steps_per_epoch <= 0:
        raise ValueError("The pretraining subset is too small for the configured batch")

    model = ViTEncoder(model_config).to(device)
    sigreg = SIGReg(
        knots=int(training_config.get("sigreg_knots", 17)),
        num_directions=int(training_config.get("sigreg_directions", 256)),
    ).to(device)
    learning_rate = float(training_config["learning_rate"])
    minimum_learning_rate = float(training_config["minimum_learning_rate"])
    if not 0 <= minimum_learning_rate <= learning_rate:
        raise ValueError("minimum_learning_rate must lie between zero and learning_rate")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    scheduler = _build_scheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        warmup_epochs=int(training_config.get("warmup_epochs", 0)),
        minimum_learning_rate=minimum_learning_rate,
    )

    output_dir = Path(
        output_dir_override or config["output"].get("directory") or (
            Path(config["output"]["scaling_directory"]) / model_config.name
            / pretraining_data_directory(config, sample_size) / f"seed_{seed}" / "pretraining"
        )
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_metadata = {
        "source": "Radio Galaxy Zoo",
        "sample_size": sample_size,
        "subset_seed": seed,
        "pretraining_pool_size": len(train_pool),
        "validation_fraction": validation_fraction,
        "validation_split_seed": split_seed,
        "train_size": len(train_records),
        "validation_size": len(validation_records),
        "validation_sample_ids": [record.sample_id for record in validation_records],
    }
    sample_ids = [record.sample_id for record in selected_records]
    last_path = output_dir / "checkpoint_last.pt"
    resume_path = Path(resume_checkpoint_override or last_path)
    start_epoch, history = resume_pretraining(
        resume_path, model=model, model_config=model_config,
        augmentation=augmentation, optimizer=optimizer, scheduler=scheduler,
        training_config=training_config, steps_per_epoch=steps_per_epoch,
        train_loader=train_loader, data_metadata=data_metadata,
        sample_ids=sample_ids, train_records=train_records, output_dir=output_dir,
    )
    if start_epoch >= epochs:
        return resume_path

    save_options = dict(model=model, model_config=model_config,
                        augmentation=augmentation.to_dict(), data_metadata=data_metadata,
                        optimizer=optimizer, scheduler=scheduler,
                        training_config=dict(training_config), steps_per_epoch=steps_per_epoch,
                        loader=train_loader, sample_ids=sample_ids)
    with tracked_run(
        config.get("wandb", {}), output_dir,
        name=f"pretrain-{model_config.name}-rgz_train_{sample_size}-val_{len(validation_records)}-seed_{seed}",
        job_type="pretraining", resume=start_epoch > 0,
        config={"stage": "pretraining", "architecture": model_config.name,
                "dataset": "RGZ20k", **dict(training_config),
                "model": asdict(model_config), "augmentation": augmentation.to_dict(),
                "data": data_metadata, "max_steps": max_steps},
    ) as run:
        return fit_pretraining(
            model, sigreg, train_loader, validation_loader, device,
            optimizer=optimizer, scheduler=scheduler, training_config=training_config,
            output_dir=output_dir, start_epoch=start_epoch, history=history,
            checkpoint_options=save_options, max_steps=max_steps,
            validation_seed=split_seed + 1, run=run,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("baseline_config.yaml")),
    )
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Actual number of RGZ training images, excluding validation.")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    checkpoint = train(
        load_config(arguments.config),
        sample_size_override=arguments.sample_size,
        model_name_override=arguments.model,
        max_steps_override=arguments.max_steps,
        output_dir_override=arguments.output_dir,
    )
    print(f"Saved RGZ checkpoint: {checkpoint}")
