import os
import sys
from yaml import safe_load


ROOT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
BIN_DIR = os.path.abspath('bin')
CONF_DIR = os.path.abspath('config')
ASSET_DIR = os.path.join(ROOT_DIR, 'assets')

# if you're running on a virtual machine (no virtual memory/page disk) this must not exceed the total amount of RAM.
GTFS_MAX_HEAP = os.environ.get('GTFS_MAX_HEAP', '2G')
OTP_MAX_HEAP = os.environ.get('OTP_MAX_HEAP', '2G')


def load_config(dir):
    with open(os.path.join(dir, 'config.yml')) as conf_file:
        config = safe_load(conf_file)
    return config
