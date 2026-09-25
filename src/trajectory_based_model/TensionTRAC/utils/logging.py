"""Small logging shim for the inference-only TensionTRAC subset."""

from __future__ import annotations

import logging
import sys


def setup_logging(output_dir=None):
    del output_dir
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name):
    return logging.getLogger(name)


def log_json_stats(stats):
    get_logger(__name__).info("json_stats: %s", stats)
