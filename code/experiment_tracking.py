"""W&B lifecycle and epoch logging shared by pretraining and classification."""

import json
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def tracked_run(settings, output_dir, *, name, job_type, config, resume=False):
    if not settings.get("enabled", False):
        yield None
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B is enabled; install wandb or set wandb.enabled: false.") from error
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "wandb_run.json"
    identity = json.loads(identity_path.read_text()) if resume and identity_path.is_file() else {}
    entity = settings.get("entity", "noe-schaerer-unige")
    project = settings.get("project", "radiolejepa")
    if identity and (identity["entity"], identity["project"]) != (entity, project):
        raise ValueError("Resuming W&B requires the original entity and project.")
    run = wandb.init(
        entity=entity, project=project, name=name, job_type=job_type,
        config=config, dir=str(output_dir.resolve()),
        mode=settings.get("mode", "online"),
        id=identity.get("id"), resume="allow" if identity else None,
    )
    try:
        identity_path.write_text(json.dumps({"id": run.id, "entity": entity,
                                            "project": project}, indent=2))
        run.config.update(config, allow_val_change=True)
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        run.define_metric("learning_rate*", step_metric="epoch")
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


def log_epoch(run, metrics):
    if run is None:
        return
    payload = {"epoch": metrics["epoch"], "learning_rate": metrics["learning_rate"]}
    for role in ("train", "validation"):
        payload.update({f"{role}/{key}": value
                        for key, value in metrics.get(role, {}).items()})
    payload.update({f"learning_rate/group_{index}": value
                    for index, value in enumerate(metrics.get("learning_rates", []))})
    run.log(payload)
