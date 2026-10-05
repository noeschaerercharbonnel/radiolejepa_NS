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
    find_fraction_evaluation, matches_mirabest_evaluation, run_directories,
)


def validation_score(metrics: dict, metric: str) -> float:
    best_epoch = int(metrics["best_epoch"])
    return float(
        next(
            row["validation"][metric]
            for row in metrics["history"]
            if int(row["epoch"]) == best_epoch
        )
    )


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


def plot_pretraining_losses(config: dict, output_dir: Path) -> Path | None:
    """Plot available SSL histories, even before downstream evaluation finishes."""
    models = list(config["scaling"]["models"])
    sizes = [int(size) for size in config["scaling"]["subset_sizes"]]
    seed = int(config["training"]["seed"])
    target = config["training"].get("epochs")
    histories = {}
    for model in models:
        for size in sizes:
            path = next((directory / "pretraining" / "metrics.json"
                         for directory in run_directories(config, model, size, seed)
                         if (directory / "pretraining" / "metrics.json").is_file()), None)
            if path is not None:
                history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
                histories[model, size] = [row for row in history
                                          if "epoch" in row and
                                          (target is None or int(row["epoch"]) <= int(target))]
    fields = [("loss", "Total LeJEPA loss"),
              ("invariance", "Invariance loss (unweighted)"),
              ("sigreg", "SIGReg loss (unweighted)")]
    if not any("loss" in row.get("train", {})
               for history in histories.values() for row in history):
        print("No saved pretraining loss history available; skipping loss plot")
        return None
    figure, axes = plt.subplots(len(models), len(fields),
                                figsize=(14, 3.2 * len(models)), squeeze=False)
    colors = {size: plt.get_cmap("tab10")(index % 10)
              for index, size in enumerate(sizes)}
    for model_index, model in enumerate(models):
        for column, (field, title) in enumerate(fields):
            axis = axes[model_index, column]
            for size in sizes:
                points = sorted((int(row["epoch"]), float(row["train"][field]))
                                for row in histories.get((model, size), [])
                                if field in row.get("train", {}) and
                                math.isfinite(float(row["train"][field])))
                if points:
                    axis.plot([epoch for epoch, _ in points],
                              [value for _, value in points], color=colors[size],
                              label=f"N={size:,}")
            axis.set(title=f"{model.replace('_patch16_224', '')}: {title}",
                     xlabel="Pretraining epoch", ylabel="Training loss")
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=8)
            else:
                axis.text(0.5, 0.5, "No saved history", ha="center", va="center",
                          transform=axis.transAxes)
    figure.suptitle(f"RGZ20k pretraining losses — seed {seed}")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pretraining_loss.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Saved pretraining loss plot: {path}")
    return path


def save_results(
    pretraining_config: dict,
    probe_config: dict,
    *,
    metric: str = "balanced_accuracy",
    evaluate_fractions: bool = True,
) -> tuple[list[dict], list[dict]]:
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
    output_dir = Path(pretraining_config["output"].get(
        "results_directory", scaling_dir.parent / "results"
    )).expanduser()
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
            probe_path = find_evaluation(pretraining_config, model, sample_size, seed)
            if probe_path is None or pretraining_path is None:
                missing.append(
                    {
                        "stage": "full_probe",
                        "model": model,
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
    plot_pretraining_losses(pretraining_config, output_dir)

    best_models = {}
    for model in models:
        sample_size = fraction_training_size(pretraining_config)
        if ((model, sample_size) in probe_metrics_by_run
                and matches_mirabest_evaluation(probe_metrics_by_run[(model, sample_size)], pretraining_config)):
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
                path = find_fraction_evaluation(pretraining_config, model, sample_size, fraction)
                if path is None:
                    missing.append(
                        {
                            "stage": "probe_fraction",
                            "model": model,
                            "sample_size": sample_size,
                            "fraction": fraction,
                            "path": str(path),
                        }
                    )
                    continue
                probe = json.loads(path.read_text(encoding="utf-8"))
                if not matches_mirabest_evaluation(probe, pretraining_config):
                    missing.append({"stage": "probe_fraction", "model": model,
                                    "fraction": fraction, "reason": "not best-validation 16k pretraining"})
                    continue
            fraction_results.append(
                {
                    "model": model,
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
                label=model.replace("_patch16_224", ""),
            )
        axes[0].set(
            title="Pretraining-data scaling",
            xlabel="RGZ20k pretraining images N",
            ylabel=metric.replace("_", " ").title(),
            ylim=(0.0, 1.0),
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
                marker="o",
                label=f"N={next((actual_sizes[(model, sample_size)] for model in models if (model, sample_size) in actual_sizes), sample_size):,}",
            )
        axes[1].set(
            title="Model-parameter scaling",
            xlabel="ViT parameters P",
            ylabel=metric.replace("_", " ").title(),
            ylim=(0.0, 1.0),
        )
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.25)
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
                label=(
                    f"{model.replace('_patch16_224', '')} "
                    f"(N={rows[0]['pretraining_sample_size']:,})"
                ),
            )
        axis.set(
            title="Frozen-probe scaling with labelled MiraBest data",
            xlabel="MiraBest training data used (%)",
            ylabel="Macro F1",
            ylim=(0.0, 1.0),
        )
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
    seed = int(config["training"]["seed"])
    metric = probe_config["scaling"]["selection_metric"]
    full_rows = [{"method": "linear", **row} for row in linear_rows]
    fraction_rows = [{"method": "linear", **row} for row in linear_fractions]
    for model in config["scaling"]["models"]:
        candidates = []
        for size in config["scaling"]["subset_sizes"]:
            path = find_evaluation(config, model, size, seed, "finetune")
            if path is None:
                missing.append({"stage": "finetune", "model": model, "sample_size": size})
                continue
            data = json.loads(path.read_text())
            full_rows.append({"method": "finetune", "model": model,
                              "sample_size": size, "seed": seed,
                              "pretraining_train_size": data.get("pretraining_data", {}).get("train_size", size),
                              "checkpoint": data["encoder_checkpoint"], **data["test"]})
            candidates.append((validation_score(data, metric), size, data))
        if not candidates:
            continue
        selected = next((item for item in candidates if int(item[1]) == fraction_training_size(config)
                         and matches_mirabest_evaluation(item[2], config)), None)
        if selected is None:
            continue
        score, size, full = selected
        for fraction in probe_config["scaling"]["train_fractions"]:
            if fraction == 1.0:
                data = full
            else:
                path = find_fraction_evaluation(config, model, size, fraction, "finetune")
                if path is None:
                    missing.append({"stage": "finetune_fraction", "model": model,
                                    "sample_size": size, "fraction": fraction, "path": str(path)})
                    continue
                data = json.loads(path.read_text())
                if not matches_mirabest_evaluation(data, config):
                    missing.append({"stage": "finetune_fraction", "model": model,
                                    "fraction": fraction, "reason": "not best-validation 16k pretraining"})
                    continue
            fraction_rows.append({"method": "finetune", "model": model,
                                  "pretraining_sample_size": size,
                                  "train_fraction": fraction,
                                  "train_samples": data["probe_data"]["split_sizes"]["train"],
                                  "selection_validation_score": score, **data["test"]})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {model: palette[index % len(palette)]
                    for index, model in enumerate(config["scaling"]["models"])}
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
                      color=model_colors[model],
                      marker="o", linestyle="-" if method == "linear" else "--",
                      label=f"{model.replace('_patch16_224', '')}: {method}")
        axis.set(xlabel="RGZ20k pretraining images" if x_key == "pretraining_train_size" else "MiraBest training fraction",
                 ylabel="Test macro F1", ylim=(0, 1))
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
    parser.add_argument("--probe-root", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    pretraining = load_config(arguments.pretraining_config)
    probe = load_config(arguments.probe_config)
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
