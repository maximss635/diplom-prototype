import logging
import os
from logging import config


def setup_logger():
    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s,%(msecs)03d]:%(name)s:%(levelname)s:%(message)s",
                "datefmt": "%a, %d %b %Y %H:%M:%S",
            },
        },
        "root": {"level": logging.DEBUG, "handlers": ["verbose_output"]},
        "handlers": {
            "verbose_output": {
                "formatter": "standard",
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)

    # Отключение логов tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
