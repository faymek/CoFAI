"""Runtime noise controls for eval entrypoints."""

from __future__ import annotations

import logging
import os
import warnings

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "urllib3",
    "filelock",
)


def suppress_unnecessary_runtime_output(*, verbose: bool = False) -> None:
    """Suppress noisy third-party logging/warnings during eval.

    This is the only public entrypoint expected by eval callers. The exact
    warning filters, environment defaults, and library-specific logger settings
    are runtime policy details kept inside this module.
    """

    _suppress_common_warnings()
    _set_quiet_env_defaults()
    _configure_library_logging(verbose=verbose)


def _suppress_common_warnings() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def _set_quiet_env_defaults() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("SUPPRESS_CUSTOM_KERNEL_WARNING", "1")


def _configure_library_logging(*, verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, force=True)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.WARNING)

    try:
        from transformers.utils import logging as hf_logging

        if verbose:
            hf_logging.set_verbosity_info()
        else:
            hf_logging.set_verbosity_error()
    except Exception:
        pass
