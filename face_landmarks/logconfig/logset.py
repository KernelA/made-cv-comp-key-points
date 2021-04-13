"""Python logging config

"""

import os
import datetime
import logging
import logging.config

from tzlocal import get_localzone


class UTCFormatter(logging.Formatter):
    LOCAL_TZ = get_localzone()

    def formatTime(self, record, datefmt=None):
        utc = datetime.datetime.fromtimestamp(record.created, UTCFormatter.LOCAL_TZ)
        return utc.isoformat(timespec="milliseconds")


LOGGER_SETUP = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default_console_thread': {
            '()': UTCFormatter,
            'format': '%(asctime)s %(levelname)s %(threadName)s %(module)s %(funcName)s %(message)s',
        },
        'default_console_process': {
            '()': UTCFormatter,
            'format': '%(asctime)s %(levelname)s %(processName)s %(module)s %(funcName)s %(message)s',
        }
    },
    'handlers': {
        'default_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default_console_thread',
            'stream': 'ext://sys.stdout'
        }
    },
    'loggers':
    {
        'kp.main': {
            'level': 'INFO',
            'handlers': ['default_console'],
            'propagate': False
        },
        'kp.dataset': {
            'level': 'INFO',
            'handlers': ['default_console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['default_console']
    }
}
