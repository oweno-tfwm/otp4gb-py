import datetime
import json
import logging
import os
import pathlib
import sys

from yaml import safe_load


ROOT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
BIN_DIR = os.path.abspath('bin')
CONF_DIR = os.path.abspath('config')
ASSET_DIR = os.path.join(ROOT_DIR, 'assets')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

# if you're running on a virtual machine (no virtual memory/page disk) this must not exceed the total amount of RAM.
PREPARE_MAX_HEAP = os.environ.get('PREPARE_MAX_HEAP', '20G')
SERVER_MAX_HEAP = os.environ.get('SERVER_MAX_HEAP', '20G')
LOG = logging.getLogger(__name__)


def load_config(dir):
    with open(os.path.join(dir, 'config.yml')) as conf_file:
        config = safe_load(conf_file)
    return config


def write_build_config(folder: pathlib.Path, date: datetime.date) -> None:
    """Load default build config values, update and write to graph folder.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to save the build config to.
    date : datetime.date
        Date of the transit data.
    """
    folder = pathlib.Path(folder)
    filename = "build-config.json"

    default_path = pathlib.Path(CONF_DIR) / filename
    if default_path.is_file():
        LOG.info("Loading default build config from: %s", default_path)
        with open(default_path, "rt") as file:
            data = json.load(file)
    else:
        data = {}

    data["transitServiceStart"] = (date - datetime.timedelta(1)).isoformat()
    data["transitServiceEnd"] =  (date + datetime.timedelta(1)).isoformat()

    config_path = folder / filename
    with open(config_path, "wt") as file:
        json.dump(data, file)
    LOG.info("Written build config: %s", config_path)
