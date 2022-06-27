
import logging
import os
import sys

from otp4gb.otp import Server


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def main():
    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error('No path provided')

    # Start OTP Server
    server = Server(opt_base_folder)
    server.start()

    input('\n\nPress any key to stop server...\n\n')

    server.stop()

if __name__ == '__main__':
    main()