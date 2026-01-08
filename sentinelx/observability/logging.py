import logging
import sys


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("sentinelx")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


logger = setup_logging()
