"""Evaluate best-validation encoders at every N and best-validation 16k encoders at reduced MB fractions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lejepa_baseline import load_config
from checkpoint_compat import (
    fraction_training_size, pretraining_data_directory, find_evaluation,
    find_fraction_evaluation, find_saved_evaluation, matches_mirabest_evaluation,
    mirabest_subset_name, run_directories,
    resolve_pretraining_steps, select_best_validation_checkpoint,
)


MODEL_COLORS = {
    "vit_tiny_patch16_224": "#1f77b4",
    "vit_small_patch16_224": "#ff7f0e",
    "vit_base_patch16_224": "#2ca02c",
    "vit_large_patch16_224": "#d62728",
}


def subset_colors(sizes):
    return {size: plt.get_cmap("tab10")(index % 10)
            for index, size in enumerate(sorted(set(sizes)))}

# center the plots
def set_score_axis_limits(axis, values):
    """Show score differences with padding and at least a ten-point range."""
    scores = [float(value) for value in values if math.isfinite(float(value))]
    if not scores:
        axis.set_ylim(0.0, 1.0)
        return
    minimum, maximum = min(scores), max(scores)
    span = min(1.0, max(0.10, 1.30 * (maximum - minimum)))
    center = (minimum + maximum) / 2.0
    lower = max(0.0, min(center - span / 2.0, 1.0 - span))
    axis.set_ylim(lower, lower + span)


def validation_score(metrics: dict, metric: str) -> float:
    best_epoch = int(metrics["best_epoch"])
    return float(
        next(
            row["validation"][metric]
            for row in metrics["history"]
            if int(row["epoch"]) == best_epoch
        )
    )


def evaluation_representation(probe_config: dict) -> str:
    representation = probe_config.get("model", {}).get("representation", "backbone")
    if representation not in ("backbone", "projector"):
        raise ValueError("model.representation must be 'backbone' or 'projector'")
    return representation


def evaluate_mirabest_fractions(pretraining_config: dict, probe_config: dict) -> list[dict]:
    """Use SSL validation winners for full MB data and best-validation 16k encoders for reduced data."""
    from run_probe_scaling import run_fraction_probe, select_fraction_checkpoint

    methods = probe_config["training"].get("methods", ["linear"])
    fractions = list(dict.fromkeys([1.0, *map(float, probe_config["scaling"]["train_fractions"])]))
    if not methods or any(method not in ("linear", "finetune") for method in methods):
        raise ValueError("methods must contain linear and/or finetune")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("MiraBest training fractions must be in (0, 1].")
    missing = []
    for model in pretraining_config["scaling"]["models"]:
        # Full MiraBest training uses the SSL validation winner at every RGZ size.
        for sample_size in pretraining_config["scaling"]["subset_sizes"]:
            for method in methods:
                try:
                    run_fraction_probe(pretraining_config, probe_config, model=model,
                                       sample_size=int(sample_size), fraction=1.0, method=method)
                except (FileNotFoundError, ValueError) as error:
                    print(f"Skipping full MiraBest {model}, n_{sample_size}: {error}")
                    missing.append({"stage": "full_probe", "model": model,
                                    "sample_size": int(sample_size), "method": method,
                                    "reason": str(error)})
        try:
            largest_size, checkpoint = select_fraction_checkpoint(pretraining_config, model)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping MiraBest fraction evaluation for {model}: {error}")
            missing.append({"stage": "best_validation_pretraining", "model": model, "sample_size": fraction_training_size(pretraining_config), "reason": str(error)})
            continue
        print(f"MiraBest fractions for {model}: using best-validation n_{largest_size} checkpoint {checkpoint}")
        for method in methods:
            for fraction in fractions:
                if fraction == 1.0:
                    continue
                run_fraction_probe(pretraining_config, probe_config, model=model,
                                   fraction=fraction, method=method)
    return missing


def _optimizer_update_count(value):
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
        return count if count >= 0 and count == float(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _pretraining_loss_points(history, split, field, *, updates=False, steps=None):
    points = []
    for row in history:
        try:
            epoch = _optimizer_update_count(row["epoch"])
            if epoch is None:
                continue
            value = float(row.get(split, {})[field])
            if updates:
                coordinate = _optimizer_update_count(row.get("optimizer_updates"))
                coordinate = coordinate if coordinate is not None else epoch * steps
            else:
                coordinate = epoch
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if coordinate >= 0 and math.isfinite(value):
            points.append((coordinate, value))
    return sorted(points)


def plot_pretraining_losses(config: dict, output_dir: Path) -> dict[str, Path]:
    """Render saved training/validation histories by epoch and optimizer update."""
    models = list(config["scaling"]["models"])
    sizes = [int(size) for size in config["scaling"]["subset_sizes"]]
    seed = int(config["training"]["seed"])
    target = config["training"].get("epochs")
    histories = {}
    update_steps = {}
    estimated_runs = set()
    for model in models:
        for size in sizes:
            path = next((directory / "pretraining" / "metrics.json"
                         for directory in run_directories(config, model, size, seed)
                         if (directory / "pretraining" / "metrics.json").is_file()), None)
            if path is not None:
                metrics = json.loads(path.read_text(encoding="utf-8"))
                history = metrics.get("history", [])
                histories[model, size] = [row for row in history
                                          if isinstance(row, dict) and
                                          _optimizer_update_count(row.get("epoch")) is not None and
                                          (target is None or int(row["epoch"]) <= int(target))]
                selected_history = histories[model, size]
                if not selected_history:
                    continue
                if all(_optimizer_update_count(row.get("optimizer_updates")) is not None
                       for row in selected_history):
                    # Counts saved per epoch need no checkpoint or configuration inference.
                    update_steps[model, size] = None
                    continue
                steps, source = resolve_pretraining_steps(metrics, path, config)
                update_steps[model, size] = steps
                if source in ("config", "saved_config"):
                    estimated_runs.add((model, size))
                    settings = "saved" if source == "saved_config" else "current"
                    print(f"Estimated updates for {model}, N={size:,}: {steps} per epoch "
                          f"from saved training size and {settings} training config; "
                          "actual per-epoch step counts and CLI step overrides are not recorded.")
                elif steps is None:
                    print(f"No usable optimizer-update count for {model}, N={size:,}; "
                          "skipping update-axis points without saved counts.")
    fields = [("loss", "Total LeJEPA loss"),
              ("invariance", "Invariance loss (unweighted)"),
              ("sigreg", "SIGReg loss (unweighted)")]
    size_colors = subset_colors(sizes)
    specifications = (
        ("train", False, "pretraining_loss.png"),
        ("validation", False, "pretraining_validation_loss.png"),
        ("train", True, "pretraining_loss_updates.png"),
        ("validation", True, "pretraining_validation_loss_updates.png"),
    )
    saved = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, updates, filename in specifications:
        series = {(model, size, field): _pretraining_loss_points(
                      histories.get((model, size), []), split, field,
                      updates=updates, steps=update_steps.get((model, size)))
                  for model in models for size in sizes for field, _ in fields}
        if not any(series.values()):
            print(f"No saved {split} losses available for "
                  f"{'optimizer updates' if updates else 'epochs'}; skipping {filename}")
            continue
        figure, axes = plt.subplots(len(models), len(fields),
                                    figsize=(14, 3.2 * len(models)), squeeze=False)
        split_title = "Training" if split == "train" else "Validation"
        has_estimates = False
        for model_index, model in enumerate(models):
            for column, (field, title) in enumerate(fields):
                axis = axes[model_index, column]
                for size in sizes:
                    points = series[model, size, field]
                    if points:
                        estimated = updates and (model, size) in estimated_runs
                        has_estimates |= estimated
                        axis.plot([coordinate for coordinate, _ in points],
                                  [value for _, value in points], color=size_colors[size],
                                  label=f"N={size:,}{'*' if estimated else ''}")
                axis.set(title=f"{model.replace('_patch16_224', '')}: {title}",
                         xlabel="Cumulative optimizer updates (epoch end)" if updates else "Pretraining epoch",
                         ylabel=f"{split_title} loss")
                axis.margins(y=0.02)
                axis.grid(alpha=0.25)
                if axis.lines:
                    axis.legend(fontsize=8)
                else:
                    axis.text(0.5, 0.5, "No saved history", ha="center", va="center",
                              transform=axis.transAxes)
        heading = (f"RGZ20k pretraining {split_title.lower()} losses — seed {seed} — "
                   f"{'optimizer updates' if updates else 'epochs'}")
        if has_estimates:
            heading += "\n* Update counts estimated from training batch sizes and step limits"
        figure.suptitle(heading)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        saved[filename] = path
        print(f"Saved pretraining loss plot: {path}")
    return saved


def plot_representation_comparison(
    config: dict, output_dir: Path, *, confident_only: bool = True,
) -> dict[str, Path]:
    """Compare saved full-MiraBest backbone/projector test F1 at best SSL epochs."""
    models = list(config["scaling"]["models"])
    sizes = sorted({int(size) for size in config["scaling"]["subset_sizes"]})
    seed = int(config["training"]["seed"])
    methods = (("linear", "probe", "linear_probe_metrics.json"),
               ("finetune", "finetune", "finetune_metrics.json"))
    scores = {method: {} for method, _, _ in methods}
    for model in models:
        for size in sizes:
            try:
                _, epoch = select_best_validation_checkpoint(config, model, size, seed)
            except (FileNotFoundError, ValueError):
                continue
            directories = run_directories(config, model, size, seed)
            for method, folder, name in methods:
                for representation in ("backbone", "projector"):
                    for directory in directories:
                        path = find_saved_evaluation(
                            directory / folder, name, representation,
                            confident_only=confident_only, pretraining_epoch=epoch,
                        )
                        if path is None:
                            continue
                        try:
                            data = json.loads(path.read_text(encoding="utf-8"))
                            score = float(data["test"]["macro_f1"])
                        except (OSError, KeyError, TypeError, ValueError):
                            continue
                        if math.isfinite(score) and 0 <= score <= 1:
                            scores[method][model, size, representation] = score
                            break

    saved = {}
    all_scores = [score for method_scores in scores.values()
                  for score in method_scores.values()]
    expected = len(models) * len(sizes) * 2
    for method, _, _ in methods:
        method_scores = scores[method]
        filename = f"{method}_representation_comparison.png"
        if not method_scores:
            print(f"No saved full-data {method} evaluations for backbone/projector comparison; "
                  f"skipping {filename}")
            continue
        unavailable = expected - len(method_scores)
        if unavailable:
            print(f"Backbone/projector comparison ({method}): {unavailable}/{expected} "
                  "evaluations unavailable; missing points are shown as gaps.")
        figure, axis = plt.subplots(figsize=(9, 5))
        for representation, linestyle, marker in (("backbone", "-", "o"),
                                                   ("projector", "--", "s")):
            for model in models:
                values = [method_scores.get((model, size, representation), math.nan)
                          for size in sizes]
                if not any(math.isfinite(value) for value in values):
                    continue
                axis.plot(sizes, values, color=MODEL_COLORS[model],
                          linestyle=linestyle, marker=marker,
                          label=f"{model.replace('_patch16_224', '')}: {representation}")
        title = "Linear probe" if method == "linear" else "Fine-tuning"
        axis.set(title=f"{title}: backbone vs projector\n100% MiraBest training data — seed {seed}",
                 xlabel="RGZ20k pretraining images N", ylabel="Test macro F1")
        axis.set_xticks(sizes, [f"{size:,}" for size in sizes])
        set_score_axis_limits(axis, all_scores)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        saved[filename] = path
        print(f"Saved representation comparison: {path}")
    return saved


def save_results(
    pretraining_config: dict,
    probe_config: dict,
    *,
    metric: str = "balanced_accuracy",
    evaluate_fractions: bool = True,
) -> tuple[list[dict], list[dict]]:
    representation = evaluation_representation(probe_config)
    confident_only = bool(probe_config["dataset"].get("confident_only", True))
    subset = mirabest_subset_name(confident_only)
    models = list(pretraining_config["scaling"]["models"])
    sample_sizes = [
        int(value) for value in pretraining_config["scaling"]["subset_sizes"]
    ]
    seed = int(pretraining_config["training"]["seed"])
    fractions = [float(value) for value in probe_config["scaling"]["train_fractions"]]
    selection_metric = str(probe_config["scaling"]["selection_metric"])
    scaling_dir = Path(
        pretraining_config["output"]["scaling_directory"]
    ).expanduser()
    results_dir = Path(pretraining_config["output"].get(
        "results_directory", scaling_dir.parent / "results"
    )).expanduser()
    output_dir = results_dir / subset / representation
    completed = []
    missing = (evaluate_mirabest_fractions(pretraining_config, probe_config)
               if evaluate_fractions else [])
    probe_metrics_by_run = {}

    for model in models:
        for sample_size in sample_sizes:
            run_dir = scaling_dir / model / pretraining_data_directory(pretraining_config, sample_size) / f"seed_{seed}"
            pretraining_path = next((directory / "pretraining" / "metrics.json"
                                     for directory in run_directories(
                                         pretraining_config, model, sample_size, seed)
                                     if (directory / "pretraining" / "metrics.json").is_file()), None)
            probe_path = find_evaluation(pretraining_config, model, sample_size, seed,
                                         representation=representation, confident_only=confident_only)
            if probe_path is None or pretraining_path is None:
                missing.append(
                    {
                        "stage": "full_probe",
                        "model": model,
                        "representation": representation,
                        "sample_size": sample_size,
                        "path": str(run_dir),
                    }
                )
                continue
            pretraining = json.loads(pretraining_path.read_text(encoding="utf-8"))
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            probe_metrics_by_run[(model, sample_size)] = probe
            completed.append(
                {
                    "model": model,
                    "representation": representation,
                    "parameter_count": int(pretraining["parameter_count"]),
                    "sample_size": sample_size,
                    "pretraining_train_size": int(pretraining.get("data", {}).get("train_size", sample_size)),
                    "seed": seed,
                    "checkpoint": probe["encoder_checkpoint"],
                    **probe["test"],
                }
            )

    completed.sort(key=lambda row: (row["model"], row["sample_size"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_pretraining_losses(pretraining_config, results_dir)
    plot_representation_comparison(pretraining_config, results_dir / subset,
                                   confident_only=confident_only)

    best_models = {}
    for model in models:
        sample_size = fraction_training_size(pretraining_config)
        if ((model, sample_size) in probe_metrics_by_run
                and matches_mirabest_evaluation(probe_metrics_by_run[(model, sample_size)], pretraining_config,
                                                confident_only=confident_only)):
            score = validation_score(probe_metrics_by_run[(model, sample_size)], selection_metric)
            best_models[model] = {
                "sample_size": sample_size,
                "validation_metric": selection_metric,
                "validation_score": score,
            }

    fraction_results = []
    for model, selection in best_models.items():
        sample_size = int(selection["sample_size"])
        for fraction in fractions:
            if fraction == 1.0:
                probe = probe_metrics_by_run[(model, sample_size)]
            else:
                path = find_fraction_evaluation(pretraining_config, model, sample_size, fraction,
                                                representation=representation, confident_only=confident_only)
                if path is None:
                    missing.append(
                        {
                            "stage": "probe_fraction",
                            "model": model,
                            "representation": representation,
                            "sample_size": sample_size,
                            "fraction": fraction,
                            "path": str(path),
                        }
                    )
                    continue
                probe = json.loads(path.read_text(encoding="utf-8"))
                if not matches_mirabest_evaluation(probe, pretraining_config, confident_only=confident_only):
                    missing.append({"stage": "probe_fraction", "model": model,
                                    "fraction": fraction, "reason": "not best-validation 16k pretraining"})
                    continue
            fraction_results.append(
                {
                    "model": model,
                    "representation": representation,
                    "pretraining_sample_size": sample_size,
                    "train_fraction": fraction,
                    "train_samples": int(probe["probe_data"]["split_sizes"]["train"]),
                    "selection_validation_score": selection["validation_score"],
                    **probe["test"],
                }
            )

    fraction_results.sort(key=lambda row: (row["model"], row["train_fraction"]))
    if "finetune" in probe_config["training"].get("methods", ["linear"]):
        collect_method_comparison(pretraining_config, probe_config, completed,
                                  fraction_results, missing, output_dir)

    if completed:
        parameter_counts = {
            row["model"]: int(row["parameter_count"]) for row in completed
        }
        values = {
            (row["model"], int(row["sample_size"])): float(row[metric])
            for row in completed
        }
        actual_sizes = {(row["model"], int(row["sample_size"])): row["pretraining_train_size"]
                        for row in completed}
        size_colors = subset_colors(sample_sizes)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for model in models:
            points = sorted(
                (size, score)
                for (candidate, size), score in values.items()
                if candidate == model
            )
            axes[0].plot(
                [actual_sizes[(model, size)] for size, _ in points],
                [score for _, score in points],
                marker="o",
                color=MODEL_COLORS[model],
                label=model.replace("_patch16_224", ""),
            )
        axes[0].set(
            title="Pretraining-data scaling",
            xlabel="RGZ20k pretraining images N",
            ylabel=metric.replace("_", " ").title(),
        )
        axes[0].legend(fontsize=8)

        for sample_size in sample_sizes:
            points = sorted(
                (parameter_counts[model], values[(model, sample_size)])
                for model in models
                if (model, sample_size) in values
            )
            axes[1].plot(
                [parameters for parameters, _ in points],
                [score for _, score in points],
                marker="o", color=size_colors[sample_size],
                label=f"N={next((actual_sizes[(model, sample_size)] for model in models if (model, sample_size) in actual_sizes), sample_size):,}",
            )
        axes[1].set(
            title="Model-parameter scaling",
            xlabel="ViT parameters P",
            ylabel=metric.replace("_", " ").title(),
        )
        axes[1].legend(fontsize=8)
        for axis in axes:
            set_score_axis_limits(axis, values.values())
            axis.grid(alpha=0.25)
        figure.suptitle(f"Frozen probe — {representation} features")
        figure.tight_layout()
        figure.savefig(
            output_dir / "full_probe_scaling.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)

    if fraction_results:
        figure, axis = plt.subplots(figsize=(7, 5))
        grouped = defaultdict(list)
        for row in fraction_results:
            grouped[row["model"]].append(row)
        for model, rows in grouped.items():
            rows.sort(key=lambda row: row["train_fraction"])
            axis.plot(
                [100 * row["train_fraction"] for row in rows],
                [row["macro_f1"] for row in rows],
                marker="o",
                color=MODEL_COLORS[model],
                label=(
                    f"{model.replace('_patch16_224', '')} "
                    f"(N={rows[0]['pretraining_sample_size']:,})"
                ),
            )
        axis.set(
            title=f"Frozen-probe scaling with labelled MiraBest data — {representation} features",
            xlabel="MiraBest training data used (%)",
            ylabel="Macro F1",
        )
        set_score_axis_limits(axis, (row["macro_f1"] for row in fraction_results))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            output_dir / "probe_fraction_f1.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)

    return completed, missing


def collect_method_comparison(config, probe_config, linear_rows, linear_fractions,
                              missing, output_dir):
    """Plot both methods without writing intermediate result tables."""
    representation = evaluation_representation(probe_config)
    confident_only = bool(probe_config["dataset"].get("confident_only", True))
    seed = int(config["training"]["seed"])
    metric = probe_config["scaling"]["selection_metric"]
    full_rows = [{"method": "linear", **row} for row in linear_rows]
    fraction_rows = [{"method": "linear", **row} for row in linear_fractions]
    for model in config["scaling"]["models"]:
        candidates = []
        for size in config["scaling"]["subset_sizes"]:
            path = find_evaluation(config, model, size, seed, "finetune",
                                   representation=representation, confident_only=confident_only)
            if path is None:
                missing.append({"stage": "finetune", "model": model, "sample_size": size})
                continue
            data = json.loads(path.read_text())
            full_rows.append({"method": "finetune", "model": model,
                              "representation": representation,
                              "sample_size": size, "seed": seed,
                              "pretraining_train_size": data.get("pretraining_data", {}).get("train_size", size),
                              "checkpoint": data["encoder_checkpoint"], **data["test"]})
            candidates.append((validation_score(data, metric), size, data))
        if not candidates:
            continue
        selected = next((item for item in candidates if int(item[1]) == fraction_training_size(config)
                         and matches_mirabest_evaluation(item[2], config, confident_only=confident_only)), None)
        if selected is None:
            continue
        score, size, full = selected
        for fraction in probe_config["scaling"]["train_fractions"]:
            if fraction == 1.0:
                data = full
            else:
                path = find_fraction_evaluation(config, model, size, fraction, "finetune",
                                                representation=representation, confident_only=confident_only)
                if path is None:
                    missing.append({"stage": "finetune_fraction", "model": model,
                                    "sample_size": size, "fraction": fraction, "path": str(path)})
                    continue
                data = json.loads(path.read_text())
                if not matches_mirabest_evaluation(data, config, confident_only=confident_only):
                    missing.append({"stage": "finetune_fraction", "model": model,
                                    "fraction": fraction, "reason": "not best-validation 16k pretraining"})
                    continue
            fraction_rows.append({"method": "finetune", "model": model,
                                  "representation": representation,
                                  "pretraining_sample_size": size,
                                  "train_fraction": fraction,
                                  "train_samples": data["probe_data"]["split_sizes"]["train"],
                                  "selection_validation_score": score, **data["test"]})
    for stem, rows in (("method_comparison", full_rows),
                       ("method_fraction_comparison", fraction_rows)):
        if not rows:
            continue
        figure, axis = plt.subplots(figsize=(9, 5))
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["model"], row["method"])].append(row)
        x_key = "pretraining_train_size" if stem == "method_comparison" else "train_fraction"
        for (model, method), group in grouped.items():
            group.sort(key=lambda row: row[x_key])
            axis.plot([row[x_key] for row in group], [row["macro_f1"] for row in group],
                      color=MODEL_COLORS[model],
                      marker="o", linestyle="-" if method == "linear" else "--",
                      label=f"{model.replace('_patch16_224', '')}: {method}")
        axis.set(xlabel="RGZ20k pretraining images" if x_key == "pretraining_train_size" else "MiraBest training fraction",
                 ylabel="Test macro F1",
                 title=f"Probe and fine-tuning — {representation} features")
        set_score_axis_limits(axis, (row["macro_f1"] for row in rows))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / f"{stem}.png", dpi=180)
        plt.close(figure)


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
        "--metric",
        choices=("accuracy", "balanced_accuracy", "macro_f1"),
        default="balanced_accuracy",
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--collect-only", action="store_true",
                        help="Plot existing results without training missing MiraBest fractions.")
    parser.add_argument("--pretraining-only", action="store_true",
                        help="Plot saved SSL training/validation histories only; run no training or downstream evaluation.")
    parser.add_argument("--probe-root", default=None)
    parser.add_argument("--representation", choices=("backbone", "projector"),
                        default=None, help="Override model.representation in the probe config.")
    parser.add_argument("--confident-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Use certain sources; --no-confident-only restores the previous selection.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    pretraining = load_config(arguments.pretraining_config)
    if arguments.pretraining_only:
        scaling_dir = Path(pretraining["output"]["scaling_directory"]).expanduser()
        output_dir = Path(pretraining["output"].get(
            "results_directory", scaling_dir.parent / "results"
        )).expanduser()
        plots = plot_pretraining_losses(pretraining, output_dir)
        print(f"Saved {len(plots)} pretraining loss figures from existing metrics")
        raise SystemExit(1 if arguments.require_complete and len(plots) != 4 else 0)
    probe = load_config(arguments.probe_config)
    if arguments.confident_only is not None:
        probe["dataset"]["confident_only"] = arguments.confident_only
    if arguments.representation is not None:
        probe.setdefault("model", {})["representation"] = arguments.representation
    if arguments.probe_root is not None:
        probe["dataset"]["root"] = arguments.probe_root
    completed, missing = save_results(
        pretraining,
        probe,
        metric=arguments.metric,
        evaluate_fractions=not arguments.collect_only,
    )
    print(f"Collected {len(completed)} full-probe runs; {len(missing)} runs are missing")
    for stage, count in Counter(run["stage"] for run in missing).items():
        print(f"  {stage}: {count} missing")
    if arguments.require_complete and missing:
        raise SystemExit(1)
