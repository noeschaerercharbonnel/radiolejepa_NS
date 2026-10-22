"""Locate outputs saved by earlier versions, including relocated checkpoints."""

import json
import pickle
from pathlib import Path


def _positive_integer(value):
    """Reject missing, fractional, or non-positive update counts."""
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
        return count if count > 0 and count == float(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _steps_from_training_metadata(data, training):
    """Infer batches for this trainer's drop_last=True, one-update-per-batch loader."""
    train_size = _positive_integer(data.get("train_size"))
    batch_size = _positive_integer(training.get("batch_size"))
    if train_size is None or batch_size is None:
        return None
    steps = train_size // batch_size
    maximum = training.get("max_steps_per_epoch")
    if maximum is not None:
        maximum = _positive_integer(maximum)
        if maximum is None:
            return None
        steps = min(steps, maximum)
    return steps if steps > 0 else None


def _pretraining_model_name(saved):
    model = saved.get("model", saved.get("model_config", {}))
    return saved.get("model_name") or (
        model.get("name") if isinstance(model, dict) else model
    )


def _checkpoint_matches_history(saved, metrics):
    """Only take counts from the same run and a checkpoint covering its history."""
    required_epoch = max(
        [_positive_integer(metrics.get("epoch")) or 0]
        + [_positive_integer(row.get("epoch")) or 0
           for row in metrics.get("history", []) if isinstance(row, dict)]
    )
    if (_positive_integer(saved.get("epoch")) or 0) < required_epoch:
        return False
    expected_model, saved_model = (
        _pretraining_model_name(metrics), _pretraining_model_name(saved)
    )
    if expected_model and expected_model != saved_model:
        return False
    expected_data, saved_data = metrics.get("data") or {}, saved.get("data") or {}
    return all(
        key in saved_data and expected_data[key] == saved_data[key]
        for key in ("train_size", "subset_seed")
        if key in expected_data
    )


def resolve_pretraining_steps(metrics, metrics_path, config) -> tuple[int | None, str]:
    """Resolve saved update counts, falling back to a labelled configuration estimate.

    Mapped checkpoint loading reads scalar metadata without eagerly materializing
    model and optimizer tensors. The configuration fallback assumes the original
    run used the supplied batch size and step limit; callers should label it.
    """
    steps = _positive_integer(metrics.get("steps_per_epoch"))
    if steps is not None:
        return steps, "metrics"

    directory = Path(metrics_path).expanduser().parent
    candidates = [directory / "checkpoint_last.pt"]
    for key in ("checkpoint", "best_checkpoint"):
        if metrics.get(key):
            candidates.append(directory / Path(metrics[key]).name)
    candidates.append(directory / "checkpoint_best_validation.pt")
    for checkpoint in dict.fromkeys(candidates):
        if not checkpoint.is_file():
            continue
        import torch

        try:
            try:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False,
                                   mmap=True)
            except TypeError as error:
                # Older PyTorch releases do not support mmap or weights_only.
                if not any(message in str(error) for message in
                           ("unexpected keyword argument", "invalid keyword argument")):
                    raise
                try:
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                except TypeError as legacy_error:
                    if not any(message in str(legacy_error) for message in
                               ("unexpected keyword argument", "invalid keyword argument")):
                        raise
                    saved = torch.load(checkpoint, map_location="cpu")
            except RuntimeError as error:
                # torch.save(..., _use_new_zipfile_serialization=False) is legacy.
                if "mmap" not in str(error) or "serialization" not in str(error):
                    raise
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, TypeError, ValueError, EOFError,
                pickle.UnpicklingError) as error:
            print(f"Could not read pretraining update metadata from {checkpoint}: {error}")
            continue
        if isinstance(saved, dict) and _checkpoint_matches_history(saved, metrics):
            steps = _positive_integer(saved.get("steps_per_epoch"))
            if steps is not None:
                return steps, "checkpoint"

    data = metrics.get("data") or {}
    steps = _steps_from_training_metadata(data, metrics.get("training_config") or {})
    if steps is not None:
        return steps, "saved_config"
    steps = _steps_from_training_metadata(data, config.get("training") or {})
    return (steps, "config") if steps is not None else (None, "unavailable")


def fraction_training_size(config):
    """Use the largest configured RGZ training count for the MB fraction study."""
    return max(int(size) for size in config["scaling"]["subset_sizes"])


def pretraining_data_directory(config, sample_size):
    """N names the actual training count, excluding held-out validation images."""
    return f"n_{int(sample_size)}"


def mirabest_subset_name(confident_only=True):
    return "MBFRConfident" if confident_only else "MBFRAll"


def matches_mirabest_subset(metrics, confident_only=True):
    """Identify old evaluations by their excluded labels when no subset is saved."""
    if not isinstance(metrics, dict) or not isinstance(metrics.get("probe_data"), dict):
        return False
    data = metrics["probe_data"]
    if data.get("subset") is not None:
        return data["subset"] == mirabest_subset_name(confident_only)
    expected = [3, 4, 7, 8, 9] if confident_only else [8, 9]
    return data.get("excluded_fine_labels") == expected


def matches_mirabest_evaluation(metrics, config, representation=None, *, confident_only=True):
    """Check that a fraction evaluation uses the largest RGZ training run."""
    data = metrics.get("pretraining_data", {})
    target = config["training"]["epochs"]
    split_seed = int(config["dataset"].get("split_seed", 14))
    return (
        matches_mirabest_subset(metrics, confident_only)
        and (representation is None or metrics.get("representation") == representation)
        and data.get("sample_size") == fraction_training_size(config)
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


def _validate_representation(representation):
    if representation not in ("backbone", "projector"):
        raise ValueError(f"Unknown representation: {representation!r}. Expected 'backbone' or 'projector'.")


def find_saved_evaluation(directory, name, representation, *, confident_only=True,
                          pretraining_epoch=None, train_fraction=1.0, require_checkpoint=False):
    """Search separate subset outputs, then compatible layouts from older versions."""
    directory = Path(directory)
    epoch_directory = (directory / f"epochs_{pretraining_epoch}"
                       if pretraining_epoch is not None else directory)
    for base in dict.fromkeys((epoch_directory, directory)):
        candidates = (
            base / representation / mirabest_subset_name(confident_only) / name,
            base / representation / name,
            base / name,
        )
        for path in candidates:
            try:
                metrics = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if (not matches_mirabest_subset(metrics, confident_only)
                    or metrics.get("representation") != representation
                    or (pretraining_epoch is not None
                        and metrics.get("pretraining_epoch") != pretraining_epoch)
                    or metrics["probe_data"].get("train_fraction", 1.0) != train_fraction):
                continue
            if require_checkpoint and not path.with_name(name.replace("_metrics.json", ".pt")).is_file():
                continue
            return path
    return None


def find_evaluation(config, model, sample_size, seed, method="linear", representation="backbone",
                    *, confident_only=True):
    _validate_representation(representation)
    try:
        _, epoch = select_best_validation_checkpoint(config, model, sample_size, seed)
    except (FileNotFoundError, ValueError):
        return None
    folder, name = (("probe", "linear_probe_metrics.json") if method == "linear"
                    else ("finetune", "finetune_metrics.json"))
    for run_dir in run_directories(config, model, sample_size, seed):
        path = find_saved_evaluation(run_dir / folder, name, representation,
                                     confident_only=confident_only, pretraining_epoch=epoch)
        if path is not None:
            return path
    return None


def find_fraction_evaluation(config, model, sample_size, fraction, method="linear", representation="backbone",
                             *, confident_only=True):
    """Find an existing MiraBest evaluation result for a particular model, pretraining dataset size, labelled training fraction, and representation."""

    _validate_representation(representation)
    try:
        _, epoch = select_best_validation_checkpoint(config, model, sample_size, config["training"]["seed"])
    except (FileNotFoundError, ValueError):
        return None
    output = config["output"]
    folder, name = (("probe_fractions", "linear_probe_metrics.json") if method == "linear"
                    else ("finetune_fractions", "finetune_metrics.json"))
    for root in [output["scaling_directory"], *output.get("legacy_directories", [])]:
        directory = (Path(root).expanduser() / folder / model / pretraining_data_directory(config, sample_size)
                     / f"seed_{config['training']['seed']}" / f"mb_train_fraction_{fraction:g}")
        path = find_saved_evaluation(directory, name, representation, confident_only=confident_only,
                                     pretraining_epoch=epoch, train_fraction=fraction)
        if path is not None:
            return path
    return None
