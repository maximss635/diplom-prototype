import json
import logging
import logging.config
import os


def read_config():
    with open("config.json", "r") as fd:
        return json.load(fd)


def setup_logger():
    level = {
        "debug": logging.DEBUG,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
    }[read_config()["logging_level"]]

    LOGGING_CONFIG = {
        "version": 1,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s,%(msecs)03d]:%(name)s:%(levelname)s:%(message)s",
                "datefmt": "%a, %d %b %Y %H:%M:%S",
            },
        },
        "root": {"level": level, "handlers": ["verbose_output"]},
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
