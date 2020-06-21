import coloredlogs
import logging
import os
import yaml

from logging.config import dictConfig

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