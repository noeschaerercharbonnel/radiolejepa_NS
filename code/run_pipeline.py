"""Run/reuse RGZ20k pretraining, then both MiraBest evaluation methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from lejepa_baseline import SUPPORTED_MODELS, load_config, train
from lejepa_probe import PROBE_REPRESENTATIONS, get_probe_representation, train_probe
from checkpoint_compat import (find_pretraining, find_evaluation, pretraining_data_directory,
                               select_best_validation_checkpoint)


def run_job(
    pretraining_config: dict,
    probe_config: dict,
    *,
    model: str,
    sample_size: int,
    seed: int,
    pretraining_max_steps: int | None = None,
    probe_max_steps: int | None = None,
    probe_max_eval_steps: int | None = None,
    checkpoint_override: str | Path | None = None,
    methods: list[str] | None = None,
    representation_override: str | None = None,
) -> Path:
    output_root = Path(
        pretraining_config["output"]["scaling_directory"]
    ).expanduser()
    run_dir = output_root / model / pretraining_data_directory(pretraining_config, sample_size) / f"seed_{seed}"
    pretraining_dir = run_dir / "pretraining"
    probe_dir = run_dir / "probe"
    representation = get_probe_representation(probe_config, representation_override)
    confident_only = bool(probe_config["dataset"].get("confident_only", True))

    skip_completed = bool(
        pretraining_config["scaling"].get("skip_completed_runs", True)
    )
    methods = methods or probe_config["training"].get("methods", ["linear"])
    if not methods or any(method not in ("linear", "finetune") for method in methods):
        raise ValueError("methods must contain linear and/or finetune")
    existing = {method: find_evaluation(pretraining_config, model, sample_size, seed, method,
                                       representation=representation, confident_only=confident_only)
                if skip_completed and checkpoint_override is None else None
                for method in methods}
    if all(existing.values()) and "epochs" not in pretraining_config["training"]:
        print(f"All requested MiraBest methods already complete: {run_dir}")
        return existing[methods[-1]]
    checkpoint = Path(checkpoint_override).expanduser() if checkpoint_override else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint_override is not None and "epochs" in pretraining_config["training"]:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.name != "checkpoint_best_validation.pt":
            raise ValueError("Full MiraBest classification requires checkpoint_best_validation.pt.")
        if (saved.get("model_config", {}).get("name") != model
                or saved.get("data", {}).get("train_size") != sample_size
                or saved.get("data", {}).get("subset_seed") != seed
                or not 0 < int(saved.get("epoch", 0)) <= int(pretraining_config["training"]["epochs"])):
            raise ValueError("Explicit best-validation checkpoint must match the requested model, N, and seed.")
        del saved
    if checkpoint is None and skip_completed:
        checkpoint = find_pretraining(pretraining_config, model, sample_size, seed)
        if checkpoint is not None:
            print(f"Reusing pretraining checkpoint: {checkpoint}")
    target = pretraining_config["training"].get("epochs")
    needs_training = checkpoint is None or not checkpoint.is_file()
    if not needs_training and target is not None and checkpoint_override is None:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        needs_training = int(saved.get("epoch", 0)) < int(target)
        if int(saved.get("epoch", 0)) > int(target):
            raise ValueError("Checkpoint exceeds configured epochs; use its epoch target or a separate output directory.")
        if not needs_training and "validation_fraction" in pretraining_config.get("dataset", {}):
            data = saved.get("data", {})
            fraction = float(pretraining_config["dataset"]["validation_fraction"])
            split_seed = int(pretraining_config["dataset"].get("split_seed", 14))
            if (data.get("validation_fraction") != fraction
                    or data.get("validation_split_seed") != split_seed
                    or data.get("sample_size") != sample_size
                    or data.get("train_size") != sample_size
                    or len(data.get("validation_sample_ids", [])) != int(20000 * fraction)):
                raise ValueError("Existing checkpoint does not use the configured fixed RGZ20k validation holdout. Use fresh output directories and clear output.legacy_directories.")
        del saved
    if needs_training:
        pretraining_config["training"]["seed"] = seed
        checkpoint = train(
            pretraining_config,
            sample_size_override=sample_size,
            model_name_override=model,
            max_steps_override=pretraining_max_steps,
            output_dir_override=checkpoint.parent if checkpoint is not None else pretraining_dir,
            resume_checkpoint_override=checkpoint,
        )
        existing = {method: None for method in methods}

    if checkpoint_override is None:
        checkpoint, _ = select_best_validation_checkpoint(
            pretraining_config, model, sample_size, seed)
    print(f"Full MiraBest classification: n_{sample_size}, best-validation encoder {checkpoint}")
    probe_config["training"]["seed"] = seed
    for method in methods:
        if existing[method] is not None:
            print(f"Reusing {method} results: {existing[method]}")
            output = existing[method]
            continue
        output = train_probe(
            probe_config,
            checkpoint_override=checkpoint,
            max_steps_override=probe_max_steps,
            max_eval_steps_override=probe_max_eval_steps,
            output_dir_override=probe_dir if method == "linear" else run_dir / "finetune",
            method=method,
            representation_override=representation,
        )
    return output


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
    parser.add_argument("--sample-size", type=int, required=True,
                        help="Actual number of RGZ training images, excluding validation.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretraining-root", default=None)
    parser.add_argument("--probe-root", default=None)
    parser.add_argument("--pretraining-max-steps", type=int, default=None)
    parser.add_argument("--probe-max-steps", type=int, default=None)
    parser.add_argument("--probe-max-eval-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Reuse a saved pretraining checkpoint without pretraining.")
    parser.add_argument("--methods", nargs="+", choices=("linear", "finetune"))
    parser.add_argument("--representation", choices=PROBE_REPRESENTATIONS,
                        help="Override model.representation (default: backbone).")
    parser.add_argument("--confident-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Use certain sources; --no-confident-only restores the previous selection.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    pretraining_config = load_config(arguments.pretraining_config)
    probe_config = load_config(arguments.probe_config)
    if arguments.confident_only is not None:
        probe_config["dataset"]["confident_only"] = arguments.confident_only
    if arguments.pretraining_root is not None:
        pretraining_config["dataset"]["batches_root"] = arguments.pretraining_root
    if arguments.probe_root is not None:
        probe_config["dataset"]["root"] = arguments.probe_root

    checkpoint = run_job(
        pretraining_config,
        probe_config,
        model=arguments.model,
        sample_size=arguments.sample_size,
        seed=arguments.seed,
        pretraining_max_steps=arguments.pretraining_max_steps,
        probe_max_steps=arguments.probe_max_steps,
        probe_max_eval_steps=arguments.probe_max_eval_steps,
        checkpoint_override=arguments.checkpoint,
        methods=arguments.methods,
        representation_override=arguments.representation,
    )
    print(f"Saved/reused MiraBest evaluation: {checkpoint}")
