import yaml
import joblib
from datetime import datetime


CONFIG_DIR = 'config/config.yml'


def load_config():
    """Function to load config files"""
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Parameters file not found in path.')
    
    return config
