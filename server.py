""" Arg0 : folder name (mandatory)
Arg1 : port number (optional)
Arg2 : verbose (optional) (OTP stdout written to this console)"""
import logging
import os
import socket
import sys


from otp4gb.otp import Server

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error("No path provided")

    try:
        port = sys.argv[2]
    except IndexError:
        port=8080

    try:
        arg3 = sys.argv[3]
        quiet=False
    except IndexError:
        quiet=True


    # Start OTP Server
    server = Server(opt_base_folder, hostname=socket.gethostname(), port=port)
    server.start( quiet )

    input("\n\n************** Press any key to stop server... **************\n\n")

    server.stop()


if __name__ == "__main__":
    main()
