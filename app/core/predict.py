"""Model loading and prediction.

Same time convention as training:

    window i = times [i, i+window-1]  ->  predicts time i+window

For n valid times and window w this yields (n - w) predictions, aligned to
the last (n - w) times of the dataset.

Models are loaded one region at a time and released right after
(clear_session), because the Render container has limited memory.
"""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)


# ============================================================
# TensorFlow (imported on demand)
# ============================================================


def _tf():
    import tensorflow as tf

    # CPU inference runs in float32. The mixed_float16 policy used for GPU
    # training only gets in the way here.
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:                                           # noqa: BLE001
        pass

    return tf


# ============================================================
# mixed_float16 saved inside the models
#
# The models were trained on a GPU under a global mixed_float16 policy, and
# Keras stores that policy per layer inside the saved file. Calling
# set_global_policy("float32") does not undo it: the saved policy wins.
#
# On CPU that is fatal rather than merely slow, because several ops have no
# float16 kernel at all. MaxPool3D is one of them:
#
#   No OpKernel was registered to support Op 'MaxPool3D' ... T=DT_HALF
#   Registered kernels: device='CPU'; T in [DT_BFLOAT16] / [DT_FLOAT]
#
# So the policy is rewritten to float32 as each layer is rebuilt. float32 is
# strictly more precise than float16, so this cannot degrade the forecast;
# it only costs memory and some speed.
# ============================================================

_dtype_patch_applied = False


def patch_keras_dtype_policy() -> None:
    """Force float32 for layers saved with a float16 policy."""
    global _dtype_patch_applied

    if _dtype_patch_applied:
        return

    from keras.src import dtype_policies

    original_get = dtype_policies.get

    def _force_float32(identifier):
        policy = original_get(identifier)
        name = getattr(policy, "name", None)
        if isinstance(name, str) and "float16" in name:
            return original_get("float32")
        return policy

    dtype_policies.get = _force_float32
    _dtype_patch_applied = True
    log.info("Patched Keras dtype policy: float16 layers will run in float32.")


# ============================================================
# Lambda layers saved by Colab
#
# The regressor models carry one `Lambda` layer whose Python function Keras
# serialized as marshalled CPython bytecode. That format is not stable across
# CPython minor versions, and the interpreter does not validate it: a code
# object from the wrong version comes back malformed and *calling* it
# segfaults the process, which no `except` can catch. One bad layer would take
# the whole service down instead of failing a single region.
#
# The models were saved on Colab's Python 3.12, which is why the Dockerfile
# pins 3.12. Verified on plain python:*-slim images, with no TensorFlow:
#
#   3.11  ops end at BINARY_SUBSCR, no RETURN_VALUE   -> segfault on call
#   3.12  identical to a locally compiled equivalent  -> returns correctly
#   3.13  decodes to unrelated opcodes                -> segfault on call
#
# The guard below turns that crash into an ordinary exception: before Keras
# builds the function, the code object is checked for the structure a callable
# function must have. Unmarshalling is safe; only calling a malformed object
# crashes. Nothing is substituted - a sound function is handed to Keras
# untouched.
# ============================================================

# Instructions that legitimately end a Python function.
_RETURN_OPS = {"RETURN_VALUE", "RETURN_CONST"}

_lambda_patch_applied = False


def _validate_code_object(code) -> None:
    """Raise if `code` is not safely callable on this interpreter."""
    import dis

    opnames = [instruction.opname for instruction in dis.get_instructions(code)]

    if opnames and opnames[-1] in _RETURN_OPS:
        return

    import io

    buffer = io.StringIO()
    try:
        dis.dis(code, file=buffer)
    except Exception:                                           # noqa: BLE001
        buffer.write("(disassembly failed)")

    raise RuntimeError(
        "A Lambda layer in this model carries bytecode that cannot be executed "
        "on this interpreter: it does not end in a return instruction, so "
        "calling it would segfault the process rather than raise.\n"
        f"Running Python {sys.version.split()[0]}; these models were saved on "
        "Colab's Python 3.12. Check the FROM line in the Dockerfile.\n"
        f"Last instructions: {opnames[-4:]}\n"
        "Disassembly:\n" + buffer.getvalue()
    )


def _guarded_func_load(code, defaults=None, closure=None, globs=None):
    """keras.src.utils.python_utils.func_load, with a sanity check first."""
    import base64
    import binascii
    import codecs
    import marshal

    if isinstance(code, (tuple, list)):
        code, defaults, closure = code
        if isinstance(defaults, list):
            defaults = tuple(defaults)

    try:
        raw = codecs.decode(code.encode("ascii"), "base64")
    except (UnicodeEncodeError, binascii.Error):
        raw = code.encode("raw_unicode_escape")

    _validate_code_object(marshal.loads(raw))

    return _original_func_load(code, defaults=defaults, closure=closure, globs=globs)


_original_func_load = None


def patch_keras_lambda_loading() -> None:
    """Install the Lambda bytecode guard."""
    global _lambda_patch_applied, _original_func_load

    if _lambda_patch_applied:
        return

    from keras.src.utils import python_utils

    _original_func_load = python_utils.func_load
    python_utils.func_load = _guarded_func_load
    _lambda_patch_applied = True
    log.debug("Lambda bytecode guard installed.")


def _cast_layer():
    tf = _tf()

    class Cast(tf.keras.layers.Layer):
        """Custom layer used when training some of the regions."""

        def call(self, inputs):
            return tf.cast(inputs, tf.float32)

    return Cast


def release_model(model=None) -> None:
    tf = _tf()
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()


# ============================================================
# Pre-processing (identical to training)
# ============================================================


def create_rolling_window(data: np.ndarray, window_size: int) -> np.ndarray:
    num_samples = len(data) - window_size
    if num_samples <= 0:
        raise ValueError(
            f"Not enough valid times: {len(data)} for a window of {window_size}."
        )
    return np.asarray([data[i:i + window_size] for i in range(num_samples)])


def fill_nans_in_blocks(data: np.ndarray, block_size: int = 100) -> np.ndarray:
    num_blocks = len(data) // block_size
    for i in range(num_blocks + 1):
        start = i * block_size
        end = min((i + 1) * block_size, len(data))
        if start >= end:
            continue
        data[start:end] = np.nan_to_num(data[start:end], nan=0.0)
    return data


def _load_stat(path, variables: list[str] | None = None) -> np.ndarray:
    """Load a saved mean/std statistic.

    These files come in two shapes, depending on how the region was trained:

      * a plain numeric array, used as saved; or
      * a pickled dict keyed by variable name (and carrying an extra
        'Flashes' entry for the target), which is turned into a vector in
        `variables` order so it broadcasts against (time, lat, lon, variable).

    Both only load with allow_pickle, which is safe here: the files are the
    project's own training artifacts from the configured Drive folder - the
    same assumption the training notebook made.
    """
    raw = np.load(path, allow_pickle=True)

    if raw.dtype != object:
        return np.squeeze(raw)

    mapping = raw.item()

    if not isinstance(mapping, dict):
        raise ValueError(f"{Path(path).name}: unexpected pickled type {type(mapping)}")

    if variables is None:
        raise ValueError(
            f"{Path(path).name} holds a dict of per-variable statistics, so the "
            f"variable order is required to build the vector."
        )

    missing = [v for v in variables if v not in mapping]
    if missing:
        raise ValueError(
            f"{Path(path).name} has no statistic for {missing}; "
            f"available keys: {sorted(mapping)}"
        )

    return np.asarray([float(mapping[v]) for v in variables], dtype=np.float32)


def _stack_variables(ds: xr.Dataset, variables: list[str]) -> np.ndarray:
    return np.stack(
        [ds[var].values for var in variables], axis=-1,
    ).astype(np.float32)


# ============================================================
# Classification (lightning probability)
# ============================================================


def predict_classification(
    ds_new: xr.Dataset,
    variables: list[str],
    model_path: Path,
    mean_path: Path,
    std_path: Path,
    window_size: int = 6,
    batch_size: int = 8,
) -> tuple[np.ndarray, int]:
    """Returns (predictions (n, lat, lon[, 1]), window actually used)."""

    tf = _tf()
    patch_keras_dtype_policy()
    Cast = _cast_layer()

    log.info("Classification: loading %s", model_path.name)

    model = tf.keras.models.load_model(
        str(model_path), compile=False, custom_objects={"Cast": Cast},
    )
    mean = np.load(mean_path, allow_pickle=True)
    std = np.load(std_path, allow_pickle=True)

    # The window comes from the model itself: each region may have its own.
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    model_window = shape[1]

    if model_window is not None and int(model_window) != window_size:
        log.info(
            "Classifier window adjusted: %d -> %d", window_size, int(model_window),
        )
        window_size = int(model_window)

    data = _stack_variables(ds_new, variables)
    rolling = create_rolling_window(data, window_size)
    rolling = fill_nans_in_blocks(rolling)
    rolling = (rolling - mean) / std

    log.info("Classification: %d windows %s", rolling.shape[0], rolling.shape[1:])

    predictions = model.predict(rolling, batch_size=batch_size, verbose=0)

    log.info(
        "Classification done: shape %s, range %.4f to %.4f",
        predictions.shape, float(np.nanmin(predictions)), float(np.nanmax(predictions)),
    )

    release_model(model)
    return predictions, window_size


# ============================================================
# Regression (flash count)
# ============================================================


def predict_regressor(
    ds_new: xr.Dataset,
    variables: list[str],
    model_path: Path,
    mean_label_path: Path,
    std_label_path: Path,
    mean_data_path: Path,
    std_data_path: Path,
    window_size: int = 5,
    batch_size: int = 8,
) -> tuple[np.ndarray, int]:
    """Returns (predictions (n, lat, lon), window actually used)."""

    tf = _tf()
    patch_keras_dtype_policy()
    patch_keras_lambda_loading()

    log.info("Regression: loading %s", model_path.name)

    model = tf.keras.models.load_model(
        str(model_path), compile=False, safe_mode=False,
    )

    mean_lbl = _load_stat(mean_label_path, variables)
    std_lbl = _load_stat(std_label_path, variables)
    mean_data = _load_stat(mean_data_path, variables)
    std_data = _load_stat(std_data_path, variables)

    # The label statistics decide the units of the output. The training
    # notebook denormalizes with these two files, but for some regions they
    # hold values near mean 0 / std 1 - the statistics of the *already
    # normalized* target - while the per-variable dict carries the raw
    # 'Flashes' scale. When that happens the output stays in normalized units
    # instead of flashes, so say so loudly rather than publish wrong numbers.
    if np.ndim(std_lbl) == 0 and abs(float(std_lbl) - 1.0) < 0.1:
        log.warning(
            "%s holds std=%.4f and mean=%.4f, which look like statistics of an "
            "already normalized target. Flash counts for this region may come "
            "out in normalized units. Confirm against the training notebook "
            "which statistic denormalizes the regressor output.",
            Path(std_label_path).name, float(std_lbl), float(np.ravel(mean_lbl)[0]),
        )

    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    model_window = shape[1]

    if model_window is not None and int(model_window) != window_size:
        log.info(
            "Regressor window adjusted: %d -> %d", window_size, int(model_window),
        )
        window_size = int(model_window)

    ds_new = ds_new.sortby("time")
    n_times = ds_new.sizes["time"]

    if n_times <= window_size:
        raise ValueError(
            f"Not enough valid times ({n_times}) for a window of {window_size}; "
            f"at least {window_size + 1} are required."
        )

    data_raw = _stack_variables(ds_new, variables)

    data_norm = (data_raw - mean_data) / (std_data + 1e-6)
    data_norm = np.asarray(data_norm, dtype=np.float32)

    if data_norm.shape != data_raw.shape:
        data_norm = data_norm.reshape(data_raw.shape)

    data_norm = np.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)

    n_windows = n_times - window_size
    windows = np.stack(
        [data_norm[i:i + window_size] for i in range(n_windows)], axis=0,
    ).astype(np.float32)

    log.info("Regression: %d windows %s", windows.shape[0], windows.shape[1:])

    preds_norm = np.squeeze(model.predict(windows, batch_size=batch_size, verbose=0))

    if preds_norm.ndim == 2:                    # only (lat, lon) left, single window
        preds_norm = preds_norm[np.newaxis, ...]

    if preds_norm.ndim != 3:
        release_model(model)
        raise ValueError(f"Unexpected prediction shape: {preds_norm.shape}")

    preds = (preds_norm * std_lbl) + mean_lbl
    preds = np.maximum(preds, 0)

    log.info(
        "Regression done: shape %s, range %.2f to %.2f",
        preds.shape, float(np.nanmin(preds)), float(np.nanmax(preds)),
    )

    release_model(model)
    return preds, window_size


# ============================================================
# Prediction -> DataArray
# ============================================================


def forecast_to_dataarray(
    forecast_data: np.ndarray,
    ds_orig: xr.Dataset,
    var_name: str,
) -> xr.DataArray:
    """Align the predictions to the last N times of the input dataset.

    Holds for any window size: N predictions -> ds.time[-N:].
    """

    ds_orig = ds_orig.sortby("time")

    latitude = ds_orig["lat"].values
    longitude = ds_orig["lon"].values

    forecast_data = np.asarray(forecast_data)

    if forecast_data.ndim == 4 and forecast_data.shape[-1] == 1:
        forecast_data = np.squeeze(forecast_data, axis=-1)

    if forecast_data.ndim != 3:
        raise ValueError(
            f"Unexpected forecast shape: {forecast_data.shape}; "
            f"expected (time, lat, lon)."
        )

    n_predictions = forecast_data.shape[0]
    all_times = ds_orig["time"].values

    if n_predictions == 0:
        raise ValueError("No prediction was generated.")

    if n_predictions > len(all_times):
        raise ValueError(
            f"More predictions ({n_predictions}) than valid times ({len(all_times)})."
        )

    expected = (len(latitude), len(longitude))
    if forecast_data.shape[1:] != expected:
        raise ValueError(
            f"Spatial mismatch: forecast {forecast_data.shape[1:]} "
            f"vs expected {expected}."
        )

    return xr.DataArray(
        forecast_data,
        coords={
            "time": all_times[-n_predictions:],
            "lat": latitude,
            "lon": longitude,
        },
        dims=["time", "lat", "lon"],
        name=var_name,
    )
