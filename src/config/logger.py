import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
_LOG_FILE = os.path.join(_LOG_DIR, "deus_bank.log")
_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _build_root_logger() -> logging.Logger:
    root = logging.getLogger("deus_bank")
    if root.handlers:
        return root  # already initialised (e.g. after uvicorn hot-reload)

    root.setLevel(_LOG_LEVEL)
    root.propagate = False

    ch = logging.StreamHandler()
    ch.setFormatter(_FORMATTER)
    root.addHandler(ch)

    os.makedirs(_LOG_DIR, exist_ok=True)
    fh = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setFormatter(_FORMATTER)
    root.addHandler(fh)

    return root


_root_logger = _build_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the deus_bank root logger.

    Args:
        name: Typically ``__name__`` from the calling module.
    """
    if name.startswith("deus_bank"):
        return logging.getLogger(name)
    return logging.getLogger(f"deus_bank.{name}")
