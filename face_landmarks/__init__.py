import logging.config

from .logconfig import LOGGER_SETUP
from .dataset import LandMarkDatset

logging.config.dictConfig(LOGGER_SETUP)
