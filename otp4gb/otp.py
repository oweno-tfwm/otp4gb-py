import atexit
import logging
import os
import subprocess
import time

import urllib.request
import urllib.parse

from otp4gb.config import BIN_DIR, PREPARE_MAX_HEAP, SERVER_MAX_HEAP

logger = logging.getLogger(__name__)
OTP_VERSION = "2.3.0"


def _java_command(heap):
    otp_jar_file = os.path.join(BIN_DIR, f"otp-{OTP_VERSION}-shaded.jar")
    return [
        "java",
        "-Xmx{}".format(heap),
        "--add-opens",
        "java.base/java.util=ALL-UNNAMED",
        "--add-opens",
        "java.base/java.io=ALL-UNNAMED",
        "-jar",
        otp_jar_file,
    ]


def prepare_graph(build_dir):
    command = _java_command(PREPARE_MAX_HEAP) + ["--build", build_dir, "--save"]
    logger.info("Running OTP build command")
    logger.debug(command)
    subprocess.run(command, check=True)


class Server:
    def __init__(self, base_dir, port=8080, hostname="localhost" ):
        self.base_dir = base_dir
        self.port = str(port)
        self.process = None
        self.hostname = hostname

    def start(self, dumpStdoutToNull=True ):
        command = _java_command(SERVER_MAX_HEAP) + [
            r"graphs\filtered",
            "--load",
            "--port",
            self.port,
            "--securePort",
            str(int(self.port)+1)
        ]
        logger.info("Starting OTP server")
        logger.debug("About to run server with %s", command)
        self.process = subprocess.Popen(
            command, cwd=self.base_dir, stdout=(subprocess.DEVNULL if dumpStdoutToNull else None)
        )
        atexit.register(lambda: self.stop())
        self._check_server()
        logger.info("OTP server started")

    def _check_server(self):
        logger.info("Checking server")
        TIMEOUT = 15
        MAX_RETRIES = 20
        server_up = False
        retries = 0
        while not server_up:
            try:
                self.send_request()
                logger.info("Server responded")
                server_up = True
            except urllib.error.URLError as error:
                if retries > MAX_RETRIES:
                    raise Exception("Maximum retries exceeded")
                retries += 1
                logger.info(
                    "Server not available. Retry %s. Server error: %s", retries, error
                )
                time.sleep(TIMEOUT)

    def send_request(self, path="", query=None):
        url = self.get_url(path, query)
        logger.debug("About to make request to %s", url)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request) as r:
            body = r.read().decode(r.info().get_param("charset") or "utf-8")
        return body

    def get_url(self, path="", query=None):
        return self.get_root_url(urllib.parse.urljoin("routers/filtered/", path), query)

    def get_root_url(self, path="", query=None):
        qs = urllib.parse.urlencode(query, safe=",:") if query else ""
        url = urllib.parse.urlunsplit(
            [
                "http",
                self.hostname + ":" + self.port,
                urllib.parse.urljoin("otp/", path),
                qs,
                None,
            ]
        )
        return url

    def stop(self):
        if not self.process or self.process.poll():
            logger.info("OTP server is not running")
            return
        logger.info("Stopping OTP server")
        self.process.terminate()
        self.process.wait(timeout=60)
        logger.info("OTP server stopped")
