import os
import sys
import tempfile
import importlib
import shutil
from easydict import EasyDict


def load_cfg(cfg_filename:str)->EasyDict:
    assert cfg_filename.endswith('.py')

    with tempfile.TemporaryDirectory() as temp_config_dir:
        temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py')
        temp_config_name = os.path.basename(temp_config_file.name)
        shutil.copyfile(cfg_filename, os.path.join(temp_config_dir, temp_config_name))
        temp_module_name = os.path.splitext(temp_config_name)[0]
        sys.path.insert(0, temp_config_dir)
        cfg = getattr(importlib.import_module(temp_module_name), 'cfg')
        assert isinstance(cfg, EasyDict)
        sys.path.pop()
        del sys.modules[temp_module_name]
        temp_config_file.close()

    return cfg