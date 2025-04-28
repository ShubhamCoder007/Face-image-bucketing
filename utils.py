import os
import yaml
import logging

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s — %(levelname)s — %(message)s'))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

cfg = load_config()
logger = setup_logger()
