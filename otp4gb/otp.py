import logging
import os
import subprocess
import time

import urllib.request
import urllib.parse

from otp4gb.config import BIN_DIR, MAX_HEAP

logger = logging.getLogger(__name__)

otp_jar_file = os.path.join(BIN_DIR, 'otp-1.5.0-shaded.jar')
otp_base = [
    'java',
    '--add-opens', 'java.base/java.util=ALL-UNNAMED',
    '--add-opens', 'java.base/java.io=ALL-UNNAMED',
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


class Server:
    def __init__(self, base_dir, port=8080):
        self.base_dir = base_dir
        self.port = str(port)

    def start(self):
        graphs_dir = os.path.join(self.base_dir, 'graphs')

        command = otp_base + [
            '--graphs', graphs_dir,
            '--router', 'filtered',
            '--port', self.port,
            '--server'
        ]
        logger.info("Starting OTP server")
        logger.debug("About to run server with %s", command)
        self.process = subprocess.Popen(command, cwd=os.getcwd())
        self._check_server()
        logger.info("OTP server started")

    def _check_server(self):
        TIMEOUT = 10
        MAX_RETRIES = 10
        server_up = False
        retries = 0
        while not server_up:
            try:
                self.send_request()
                server_up = True
            except urllib.error.URLError:
                if retries > MAX_RETRIES:
                    raise Exception('Maximum retries exceeded')
                retries += 1
                logger.info('Server not available. Retry %s', retries)
                time.sleep(TIMEOUT)

    def send_request(self):
        url = urllib.parse.urlunparse([
          'http',
          'localhost:' + self.port,
          '/otp/routers/filtered/',
          None,
          None,
          None,
        ])
        logger.debug("About to make request to %s", url)
        # url = 'http://localhost:8080/otp/routers/filtered/'
        request = urllib.request.Request(url)
        urllib.request.urlopen(request)

    def stop(self):
        logger.info("Stopping OTP server")
        self.process.terminate()
        self.process.wait(timeout=1)
        logger.info("OTP server stopped")
