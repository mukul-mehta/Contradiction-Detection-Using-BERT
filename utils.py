import datetime
import logging
import os
import time
from configparser import ConfigParser
from logging.config import dictConfig

import coloredlogs
import yaml


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def read_config(filename="config.ini", section=""):
    if not section:
        raise Exception("Section not specified")

    parser = ConfigParser()
    parser.optionxform = str
    parser.read(filename)

    config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            config[param[0]] = param[1]
    else:
        raise Exception(
            "Section {0} not found in the {1} file".format(section, filename)
        )
    return config


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
