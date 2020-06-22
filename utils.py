import coloredlogs
import datetime
import logging
import os
import time
import yaml

from logging.config import dictConfig


def format_time(elapsed):
  elapsed_rounded = int(round(elapsed))
  return str(datetime.timedelta(seconds = elapsed_rounded))

class LogUtils:

    @staticmethod
    def setup_logger(name, default_path='logging.yaml', default_level=logging.DEBUG):
        path = default_path
        if os.path.exists(path):
            with open(path, 'rt') as f:
                try:
                    config = yaml.safe_load(f.read())
                    dictConfig(config)
                    coloredlogs.install(level=default_level)
                except Exception as e:
                    print(e)
                    print('Error in Logging Configuration. Using default configs')
                    logging.basicConfig(level=default_level)
                    coloredlogs.install(level=default_level)
        else:
            logging.basicConfig(level=default_level)
            coloredlogs.install(level=default_level)
            print('Failed to load configuration file. Using default configs')
        return logging.getLogger(name)

LOG = LogUtils.setup_logger(__name__)