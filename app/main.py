"""Entry point of the lightning prediction API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.config import settings
from app.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    log.info("Data directory: %s", settings.data_dir)
    start_scheduler(settings)
    yield
    stop_scheduler()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Daily operational lightning forecast over South America from the "
        "0.25 deg GFS, using per-region ConvLSTM models (classification plus "
        "regression), together with the GFS precipitation fields."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")
