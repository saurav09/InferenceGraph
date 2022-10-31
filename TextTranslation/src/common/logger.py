import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(funcName)20s() - %(levelname)s - %(message)s")
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = current_dir + "/../translation.log"
APP_NAME = 'InferenceGraph'


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', backupCount=90)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger():
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
