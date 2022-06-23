import logging
import os
import subprocess

from otp4gb.config import BIN_DIR, MAX_HEAP

logger = logging.getLogger(__name__)

otp_jar_file = os.path.join(BIN_DIR, 'otp-1.5.0-shaded.jar')
otp_base = [
    'java',
    '--add-opens', 'java.base/java.util=ALL-UNNAMED',
    '--add-opens', 'java.base/java.io=ALL-UNNAMED'
    '-Xmx{}'.format(MAX_HEAP),
    '-jar', otp_jar_file
]


def prepare_graph(build_dir):
    command = otp_base + [
        "--build", build_dir
    ]
    logger.info('Running OTP build command')
    logger.debug(command)
    subprocess.run(' '.join(command), shell=True)


def run_server(base_dir):
    graphs_dir = os.path.join(base_dir, 'graphs')
    router_dir = os.path.join(graphs_dir, 'filtered')

    command = otp_base + [
        '--graphs', graphs_dir,
        '--router', router_dir,
        '--server'
    ]
