"""Locate outputs saved by earlier versions, including relocated checkpoints."""

import json
from pathlib import Path

def fraction_training_size(config):
    """Use the largest configured RGZ training count for the MB fraction study."""
    return max(int(size) for size in config["scaling"]["subset_sizes"])


def pretraining_data_directory(config, sample_size):
    """N names the actual training count, excluding held-out validation images."""
    return f"n_{int(sample_size)}"


def matches_mirabest_evaluation(metrics, config):
    """Check that a fraction evaluation uses the largest RGZ training run."""
    data = metrics.get("pretraining_data", {})
    target = config["training"]["epochs"]
    split_seed = int(config["dataset"].get("split_seed", 14))
    return (
        data.get("sample_size") == fraction_training_size(config)
        and data.get("train_size") == fraction_training_size(config)
        and data.get("validation_size") == 4000
        and len(set(data.get("validation_sample_ids", []))) == 4000
        and data.get("validation_split_seed") == split_seed
        and data.get("subset_seed") == int(config["training"]["seed"])
        and Path(metrics.get("encoder_checkpoint", "")).name == "checkpoint_best_validation.pt"
        and 0 < int(metrics.get("pretraining_epoch") or 0) <= int(target)
    )


def run_directories(config, model, sample_size, seed):
    output = config["output"]
    roots = [output["scaling_directory"], *output.get("legacy_directories", [])]
    return [Path(root).expanduser() / model / pretraining_data_directory(config, sample_size) / f"seed_{seed}"
            for root in roots]


def resolve_checkpoint(pretraining_dir, metrics=None):
    """Prefer the recorded checkpoint, then recover a relocated file by name."""
    directory = Path(pretraining_dir)
    candidates = []
    for key in ("checkpoint", "best_checkpoint"):
        value = (metrics or {}).get(key)
        if value:
            recorded = Path(value).expanduser()
            candidates.extend([recorded, directory / recorded.name])
    candidates.extend(directory / name for name in
                      ("checkpoint_best.pt", "checkpoint_last.pt"))
    return next((path for path in candidates if path.is_file()), None)


def find_pretraining(config, model, sample_size, seed):
    for run_dir in run_directories(config, model, sample_size, seed):
        metrics_path = run_dir / "pretraining" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        # Without final metrics, a last checkpoint may be an interrupted run.
        last = run_dir / "pretraining" / "checkpoint_last.pt"
        checkpoint = last if last.is_file() else resolve_checkpoint(run_dir / "pretraining", metrics)
        if checkpoint is not None:
            return checkpoint
    return None


def select_best_validation_checkpoint(config, model, sample_size, seed):
    """Select the lowest-validation-loss checkpoint for this model, N, and seed."""
    import torch

    for directory in run_directories(config, model, sample_size, seed):
        checkpoint = directory / "pretraining" / "checkpoint_best_validation.pt"
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            return checkpoint, int(saved["epoch"])
    raise FileNotFoundError(f"No best-validation checkpoint for {model}, n_{sample_size}, seed_{seed}.")


def find_evaluation(config, model, sample_size, seed, method="linear"):
    try:
        _, epoch = select_best_validation_checkpoint(config, model, sample_size, seed)
    except (FileNotFoundError, ValueError):
        return None
    folder, name = (("probe", "linear_probe_metrics.json") if method == "linear"
                    else ("finetune", "finetune_metrics.json"))
    for run_dir in run_directories(config, model, sample_size, seed):
        path = run_dir / folder / f"epochs_{epoch}" / "projector" / name
        if path.is_file():
            return path
    return None


def find_fraction_evaluation(config, model, sample_size, fraction, method="linear"):
    try:
        _, epoch = select_best_validation_checkpoint(config, model, sample_size, config["training"]["seed"])
    except (FileNotFoundError, ValueError):
        return None
    output = config["output"]
    folder, name = (("probe_fractions", "linear_probe_metrics.json") if method == "linear"
                    else ("finetune_fractions", "finetune_metrics.json"))
    for root in [output["scaling_directory"], *output.get("legacy_directories", [])]:
        path = (Path(root).expanduser() / folder / model / pretraining_data_directory(config, sample_size)
                / f"seed_{config['training']['seed']}" / f"mb_train_fraction_{fraction:g}"
                / f"epochs_{epoch}" / "projector" / name)
        if path.is_file():
            return path
    return None
