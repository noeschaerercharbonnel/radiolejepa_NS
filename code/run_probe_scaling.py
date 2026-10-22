"""Evaluate RGZ20k-pretrained ViTs with different labelled MiraBest fractions."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from lejepa_baseline import SUPPORTED_MODELS, load_config
from lejepa_probe import PROBE_REPRESENTATIONS, get_probe_representation, train_probe
from checkpoint_compat import (
    fraction_training_size, find_evaluation, pretraining_data_directory,
    find_fraction_evaluation, matches_mirabest_evaluation,
    select_best_validation_checkpoint,
)


def select_fraction_checkpoint(pretraining_config: dict, model: str) -> tuple[int, Path]:
    """Use this model type's best-validation encoder at the largest RGZ size."""
    sample_size = fraction_training_size(pretraining_config)
    checkpoint, _ = select_best_validation_checkpoint(
        pretraining_config, model, sample_size,
        int(pretraining_config["training"]["seed"]),
    )
    return sample_size, checkpoint


def run_fraction_probe(
    pretraining_config: dict,
    probe_config: dict,
    *,
    model: str,
    fraction: float,
    method: str = "linear",
    sample_size: int | None = None,
    representation_override: str | None = None,
) -> Path:
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0, 1].")
    representation = get_probe_representation(probe_config, representation_override)
    confident_only = bool(probe_config["dataset"].get("confident_only", True))
    seed = int(pretraining_config["training"]["seed"])
    largest_size = fraction_training_size(pretraining_config)
    sample_size = largest_size if sample_size is None else sample_size
    if fraction < 1.0 and sample_size != largest_size:
        raise ValueError(f"Reduced MiraBest fractions require the largest RGZ training run: n_{largest_size}.")
    checkpoint, _ = select_best_validation_checkpoint(pretraining_config, model, sample_size, seed)
    if probe_config.get("scaling", {}).get("skip_completed_runs", True):
        existing = (find_evaluation(pretraining_config, model, sample_size, seed, method,
                                    representation=representation, confident_only=confident_only)
                    if fraction == 1.0 else
                    find_fraction_evaluation(pretraining_config, model, sample_size, fraction, method,
                                             representation=representation, confident_only=confident_only))
        if existing is not None and (fraction == 1.0 or matches_mirabest_evaluation(
                json.loads(existing.read_text()), pretraining_config, representation,
                confident_only=confident_only)):
            print(f"Reusing {method} fraction results: {existing}")
            return existing
    output_root = Path(pretraining_config["output"]["scaling_directory"])
    output_dir = (
        output_root
        / ("probe_fractions" if method == "linear" else "finetune_fractions")
        / model
        / pretraining_data_directory(pretraining_config, sample_size)
        / f"seed_{seed}"
        / f"mb_train_fraction_{fraction:g}"
    )
    if fraction == 1.0:
        output_dir = output_root / model / pretraining_data_directory(pretraining_config, sample_size) / f"seed_{seed}" / (
            "probe" if method == "linear" else "finetune")
    print(
        f"model={model} rgz_train_images={sample_size} checkpoint={checkpoint} "
        f"probe_fraction={fraction:g} representation={representation}"
    )
    run_config = deepcopy(probe_config)
    run_config["training"]["seed"] = seed
    # Completed results were checked above; prevent the trainer reusing stale ones.
    run_config.setdefault("scaling", {})["skip_completed_runs"] = False
    return train_probe(
        run_config,
        checkpoint_override=checkpoint,
        output_dir_override=output_dir,
        train_fraction_override=fraction,
        method=method,
        representation_override=representation,
    )


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
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--sample-size", type=int, default=None,
                        help="RGZ training count for fraction 1.0; defaults to the largest configured size.")
    parser.add_argument("--probe-root", default=None)
    parser.add_argument("--methods", nargs="+", choices=("linear", "finetune"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--representation", choices=PROBE_REPRESENTATIONS,
                        help="Override model.representation (default: backbone).")
    parser.add_argument("--confident-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Use certain sources; --no-confident-only restores the previous selection.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    pretraining = load_config(arguments.pretraining_config)
    probe = load_config(arguments.probe_config)
    if arguments.confident_only is not None:
        probe["dataset"]["confident_only"] = arguments.confident_only
    if arguments.probe_root is not None:
        probe["dataset"]["root"] = arguments.probe_root
    if arguments.seed is not None:
        pretraining["training"]["seed"] = arguments.seed
        probe["training"]["seed"] = arguments.seed
    for method in arguments.methods or probe["training"].get("methods", ["linear"]):
        run_fraction_probe(pretraining, probe, model=arguments.model,
                           fraction=arguments.fraction, method=method,
                           sample_size=arguments.sample_size,
                           representation_override=arguments.representation)
