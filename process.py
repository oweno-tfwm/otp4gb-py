import logging
import os
import sys
from otp4gb.otp import Server

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    try:
        opt_base_folder = os.path.abspath(sys.argv[1])
    except IndexError:
        logger.error('No path provided')

    # Start OTP Server
    server = Server(opt_base_folder)
    server.start()

    # Load Northern MSOAs

    # Filter MSOAs by bounding box

    # For each filtered MSOA add to Origins and Destinations

    # Define acceptable modes

    # Create output directory

    # Calculate ISOChrones from Bradford (as a debug)

    # For each destination
    #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
    #   Write isochrone
    #   Calculate all possible origins within travel time by minutes
    #

    # Stop OTP Server
    server.stop()


if __name__ == '__main__':
    main()
