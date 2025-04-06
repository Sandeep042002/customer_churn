import logging
import os
import sys
import joblib

logging.basicConfig(
    format="%(asctime)s — %(name)s:%(lineno)d — %(levelname)s — %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def get_logger(module_name: str):
    logging.basicConfig(
        format="%(asctime)s — %(name)s:%(lineno)d — %(levelname)s — %(message)s",
        level=logging.INFO,
    )

    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    return logger


SAVE_DIR = "artifacts"


def save_model(obj, filename):

    filepath = os.path.join(SAVE_DIR, filename)
    logger.info(f"save in the {obj}")
    joblib.dump(obj, filepath)


def load_model(filepath):
    """
    Load a saved model or transformer from the given relative path.
    """
    if not os.path.isabs(filepath):
        filepath = (
            os.path.join(SAVE_DIR, filepath)
            if not filepath.startswith(SAVE_DIR)
            else filepath
        )

    if not os.path.exists(filepath):
        filename = os.path.basename(filepath)
        raise FileNotFoundError(f"❌ File '{filename}' not found in '{filepath}'")

    return joblib.load(filepath)
