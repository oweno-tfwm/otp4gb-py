import logging
import sys

FORMAT_STRING = '%(asctime)s.%(msecs)03d:%(levelname)s:%(processName)s:%(module)s:%(message)s'

logging.basicConfig(level=logging.INFO,
                    format=FORMAT_STRING,
                    datefmt='%H:%M:%S')

flat_formatter = logging.Formatter(FORMAT_STRING, '%H:%M:%S')

stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.setFormatter(flat_formatter)

def get_logger(*args, **kwargs):
    return logging.getLogger(*args, **kwargs)
