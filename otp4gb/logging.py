import datetime
import logging
import os
import pathlib
import sys


ROOT_DIR = pathlib.Path().absolute()
LOG_DIR = ROOT_DIR / "logs"


FORMAT_STRING = (
    "%(asctime)s.%(msecs)03d:%(levelname)s:%(processName)s:%(module)s:%(message)s"
)


flat_formatter = logging.Formatter(FORMAT_STRING, "%H:%M:%S")

stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.setFormatter(flat_formatter)


def file_handler_factory(filename, dir=LOG_DIR, level=logging.INFO):
    if not os.path.exists(dir):
        os.makedirs(dir)
    handler = logging.FileHandler(os.path.join(dir, filename), mode="w")
    handler.setFormatter(flat_formatter)
    handler.setLevel(level)
    return handler

def get_logger(*args, **kwargs):
    logger = logging.getLogger(*args, **kwargs)
    return logger
 
def configure_app_logging(consoleLevel=logging.INFO, fileLevel=None, dir=LOG_DIR):
    """Sets root level logging, creates ./logs folder with file logger writing to it:
    file level logging defaults to same as console"""

    if fileLevel is None:
        fileLevel = consoleLevel

    logging.basicConfig(format=FORMAT_STRING, datefmt="%H:%M:%S", level=consoleLevel)

    #remember logging is thread-safe but NOT process safe.
    #https://realpython.com/python-logging-source-code/ 
    #https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    
    log_file = file_handler_factory(
        f"process-{datetime.date.today():%Y%m%d}-pid-{os.getpid()}.log", dir=dir, level=fileLevel
    )

    #add file handler to root logger
    logging.getLogger().addHandler( log_file )



