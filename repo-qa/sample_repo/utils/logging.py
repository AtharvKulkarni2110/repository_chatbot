"""Simple structured logging helper used across the app."""

import json
import sys
from datetime import datetime


def log_event(level: str, message: str, **fields):
    """Print a JSON log line to stdout with a timestamp and level."""
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level.upper(),
        "message": message,
        **fields,
    }
    print(json.dumps(record), file=sys.stdout)


def log_info(message: str, **fields):
    log_event("info", message, **fields)


def log_error(message: str, **fields):
    log_event("error", message, **fields)
