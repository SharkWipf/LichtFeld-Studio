# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""
W&B logger for LichtFeld Studio training.

Usage (headless):
  ./build/LichtFeld-Studio --train ... --python-script src/python/sample/wandb_logger.py

Configuration via env vars (all optional):
  LFS_WANDB_PROJECT / WANDB_PROJECT
  LFS_WANDB_ENTITY  / WANDB_ENTITY
  LFS_WANDB_NAME    / WANDB_NAME
  LFS_WANDB_GROUP   / WANDB_GROUP
  LFS_WANDB_TAGS    / WANDB_TAGS (comma-separated)
  LFS_WANDB_MODE    / WANDB_MODE (e.g. online|offline|disabled)
  LFS_WANDB_LOG_EVERY (default: 1)
"""

from __future__ import annotations

import os
import time

import wandb
import lichtfeld as lf


_run = None
_start_time = None
_last_step = None
_log_every = max(1, int(os.getenv("LFS_WANDB_LOG_EVERY", "1")))


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _init_run() -> None:
    global _run, _start_time
    if _run is not None:
        return

    project = _env("LFS_WANDB_PROJECT") or _env("WANDB_PROJECT") or "lichtfeld"
    entity = _env("LFS_WANDB_ENTITY") or _env("WANDB_ENTITY")
    name = _env("LFS_WANDB_NAME") or _env("WANDB_NAME")
    group = _env("LFS_WANDB_GROUP") or _env("WANDB_GROUP")
    mode = _env("LFS_WANDB_MODE") or _env("WANDB_MODE")

    tags_raw = _env("LFS_WANDB_TAGS") or _env("WANDB_TAGS")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else None

    ctx = lf.context()
    config = {
        "max_iterations": ctx.max_iterations,
        "strategy": ctx.strategy,
    }

    _run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=tags,
        mode=mode,
        config=config,
    )

    wandb.define_metric("iter")
    wandb.define_metric("train/*", step_metric="iter")

    _start_time = time.time()


def _log_stats(force: bool = False) -> None:
    global _last_step
    ctx = lf.context()

    if not force and ctx.iteration % _log_every != 0:
        return

    if _last_step == ctx.iteration and not force:
        return

    _init_run()

    try:
        lr = lf.session().optimizer().get_lr()
    except Exception:
        lr = None

    try:
        g = lf.gaussians()
        sh_degree = g.sh_degree
        max_sh_degree = g.max_sh_degree
    except Exception:
        sh_degree = None
        max_sh_degree = None

    elapsed = None
    if _start_time is not None:
        elapsed = time.time() - _start_time

    payload = {
        "iter": ctx.iteration,
        "train/loss": ctx.loss,
        "train/lr": lr,
        "train/num_gaussians": ctx.num_gaussians,
        "train/is_refining": int(ctx.is_refining),
        "train/phase": ctx.phase,
        "train/strategy": ctx.strategy,
    }

    if sh_degree is not None:
        payload["train/sh_degree"] = sh_degree
    if max_sh_degree is not None:
        payload["train/max_sh_degree"] = max_sh_degree
    if elapsed is not None:
        payload["train/elapsed_sec"] = elapsed

    wandb.log(payload, step=ctx.iteration)
    _last_step = ctx.iteration


@lf.on_training_start
def _on_training_start(_event: dict) -> None:
    _init_run()
    _log_stats(force=True)


@lf.on_post_step
def _on_post_step(_event: dict) -> None:
    _log_stats(force=False)


@lf.on_training_end
def _on_training_end(_event: dict) -> None:
    _log_stats(force=True)
    if _run is not None:
        wandb.finish()
