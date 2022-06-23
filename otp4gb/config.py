import os


ROOT_DIR = os.getcwd()
BIN_DIR = os.path.abspath('bin')
CONF_DIR = os.path.abspath('config')

# if you're running on a virtual machine (no virtual memory/page disk) this must not exceed the total amount of RAM. 80G = 80 gigabytes.
MAX_HEAP = os.environ.get('MAX_HEAP', '80G')
