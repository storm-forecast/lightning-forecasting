"""Dependencies shared by the routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import Settings, get_settings
from app.core import state


def settings_dep() -> Settings:
    return get_settings()


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guards the admin endpoints."""

    cfg = get_settings()

    if not cfg.api_key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Admin endpoints are disabled: set API_KEY.",
        )

    if x_api_key != cfg.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key.")


def resolve_run(run_id: str | None = None) -> str:
    """Resolve the run id; 'latest' or empty means the most recent run."""

    cfg = get_settings()

    if run_id and run_id != "latest":
        if state.get_manifest(cfg.products_dir, run_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found.")
        return run_id

    latest = state.get_latest_run_id(cfg.products_dir)

    if latest is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No run available yet. Trigger POST /api/v1/admin/run.",
        )

    return latest
