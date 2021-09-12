import yaml
from easydict import EasyDict as edict


def load_config(config_file):
    """
    """
    with open(config_file) as f:
        config = yaml.load(f, yaml.FullLoader)
    config = edict(config)
    return config