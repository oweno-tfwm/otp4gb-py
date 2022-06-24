import os
import sys


ROOT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
BIN_DIR = os.path.abspath('bin')
CONF_DIR = os.path.abspath('config')
ASSET_DIR = os.path.join(ROOT_DIR, 'assets')

# if you're running on a virtual machine (no virtual memory/page disk) this must not exceed the total amount of RAM. 80G = 80 gigabytes.
MAX_HEAP = os.environ.get('MAX_HEAP', '80G')
