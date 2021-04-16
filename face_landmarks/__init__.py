import logging.config

from .logconfig import LOGGER_SETUP
from .dataset import LandMarkDatset, SquarePaddingResize, normalize_landmarks, denormalize_landmarks

logging.config.dictConfig(LOGGER_SETUP)
