"""Run state and catalogue of the generated products.

On-disk layout (PRODUCTS_DIR):

    2026102300/
        manifest.json
        lightning/
            gfs_prediction_<region>.nc
            gfs_prediction_ALL_regions_max.nc
            gfs_prediction_ALL_regions_suavisado.nc
        precipitation/
            gfs_precipitation.nc
            gfs_precipitation_suavisado.nc
    latest.json          -> points at the most recently completed run
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
LATEST_NAME = "latest.json"

LIGHTNING_SUBDIR = "lightning"
PRECIP_SUBDIR = "precipitation"

_lock = threading.Lock()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_id_for(run_date: datetime) -> str:
    """Run identifier: run date plus run hour (always 00z)."""
    return f"{run_date:%Y%m%d}00"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("Invalid JSON, ignoring: %s", path)
        return None


# ============================================================
# Run state
# ============================================================


class RunState:
    """State of the current/last run, persisted in state.json."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        return read_json(self.path) or {
            "status": "idle",
            "run_id": None,
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
            "steps": [],
        }

    def update(self, **fields: Any) -> dict[str, Any]:
        with _lock:
            state = self.read()
            state.update(fields)
            state["updated_at"] = utcnow_iso()
            write_json_atomic(self.path, state)
            return state

    def start(self, run_id: str) -> dict[str, Any]:
        return self.update(
            status="running",
            run_id=run_id,
            started_at=utcnow_iso(),
            finished_at=None,
            duration_seconds=None,
            error=None,
            steps=[],
        )

    def step(self, message: str) -> None:
        with _lock:
            state = self.read()
            steps = state.get("steps") or []
            steps.append({"at": utcnow_iso(), "message": message})
            state["steps"] = steps[-40:]
            state["updated_at"] = utcnow_iso()
            write_json_atomic(self.path, state)
        log.info("[step] %s", message)

    def finish(self, run_id: str) -> dict[str, Any]:
        state = self.read()
        started = state.get("started_at")
        duration = None
        if started:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(started)
            duration = round(delta.total_seconds(), 1)
        return self.update(
            status="ok",
            run_id=run_id,
            finished_at=utcnow_iso(),
            duration_seconds=duration,
            error=None,
        )

    def fail(self, run_id: str | None, error: str) -> dict[str, Any]:
        return self.update(
            status="error",
            run_id=run_id,
            finished_at=utcnow_iso(),
            error=error,
        )

    @property
    def is_running(self) -> bool:
        return self.read().get("status") == "running"


# ============================================================
# Product catalogue
# ============================================================


def run_dir(products_dir: Path, run_id: str) -> Path:
    return Path(products_dir) / run_id


def manifest_path(products_dir: Path, run_id: str) -> Path:
    return run_dir(products_dir, run_id) / MANIFEST_NAME


def save_manifest(products_dir: Path, run_id: str, manifest: dict[str, Any]) -> Path:
    path = manifest_path(products_dir, run_id)
    write_json_atomic(path, manifest)
    return path


def set_latest(products_dir: Path, run_id: str) -> None:
    write_json_atomic(
        Path(products_dir) / LATEST_NAME,
        {"run_id": run_id, "updated_at": utcnow_iso()},
    )


def get_latest_run_id(products_dir: Path) -> str | None:
    pointer = read_json(Path(products_dir) / LATEST_NAME)
    if pointer and pointer.get("run_id"):
        run_id = str(pointer["run_id"])
        if manifest_path(products_dir, run_id).exists():
            return run_id

    # Fallback: the most recent run that has a manifest on disk.
    runs = list_runs(products_dir)
    return runs[0] if runs else None


def list_runs(products_dir: Path) -> list[str]:
    products_dir = Path(products_dir)
    if not products_dir.exists():
        return []
    runs = [
        item.name
        for item in products_dir.iterdir()
        if item.is_dir() and (item / MANIFEST_NAME).exists()
    ]
    return sorted(runs, reverse=True)


def get_manifest(products_dir: Path, run_id: str) -> dict[str, Any] | None:
    return read_json(manifest_path(products_dir, run_id))


def prune_runs(products_dir: Path, keep: int) -> list[str]:
    """Remove the oldest runs, keeping the `keep` most recent ones."""

    runs = list_runs(products_dir)
    removed: list[str] = []

    for run_id in runs[keep:]:
        target = run_dir(products_dir, run_id)
        try:
            shutil.rmtree(target)
            removed.append(run_id)
            log.info("Old run removed: %s", run_id)
        except OSError as error:
            log.warning("Could not remove %s: %s", target, error)

    return removed


def resolve_product(
    products_dir: Path,
    run_id: str,
    relative: str,
) -> Path:
    """Resolve a product path, refusing to escape the run directory."""

    base = run_dir(products_dir, run_id).resolve()
    target = (base / relative).resolve()

    if not str(target).startswith(str(base) + os.sep):
        raise ValueError("Path outside the run directory.")
    if not target.is_file():
        raise FileNotFoundError(relative)

    return target
