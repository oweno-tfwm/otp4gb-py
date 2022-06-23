import getopt
import logging
import os
import shutil
import sys
from otp4gb.gtfs_filter import filter_gtfs_files

from otp4gb.osmconvert import osm_convert
from otp4gb.config import CONF_DIR
from otp4gb.otp import prepare_graph
from otp4gb.bounds import bounds

logging.basicConfig(level=logging.DEBUG)


config = {}


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'F',
                                   ['force'])
    except getopt.GetoptError as err:
        logging.error(err)
        sys.exit(2)
    opt_force = False
    opt_base_folder = os.path.abspath('Assets')
    for o, a in opts:
        if o == '-F' or o == '--force':
            opt_force = True
        else:
            assert False, "Unhandled option"

    input_dir = os.path.join(opt_base_folder, 'input')

    filtered_graph_folder = os.path.join(opt_base_folder, 'graphs', 'filtered')

    date_filter_string = '2019-09-10:2019-09-11'

    extents = bounds.get('bradford')

    if opt_force:
        shutil.rmtree(filtered_graph_folder, ignore_errors=True)

    if os.path.exists(filtered_graph_folder):
        logging.warning(
            'A folder of filtered GTFS and OSM files already exists. To filter again, delete this folder.')
    else:
        os.makedirs(filtered_graph_folder)

        # We need to crop the osm.pbf file and all of the GTFS public transport files for GB
        # And then put them all in one folder, which we then use to run an open trip planner instance.
        # see https://github.com/odileeds/ATOCCIF2GTFS for timetable files
        # These need to reside in base_folder/input/gtfs
        filter_gtfs_files(os.path.join(input_dir, 'gtfs'),
                          output_dir=filtered_graph_folder,
                          date=date_filter_string,
                          extents=extents)

        # Crop the osm.pbf map of GB to the bounding box
        # If you are not using Windows a version of osmconvert on your platform may be available via https://wiki.openstreetmap.org/wiki/Osmconvert
        osm_convert(os.path.join(input_dir, 'great-britain-latest.osm.pbf'),
                    os.path.join(filtered_graph_folder, 'gbfiltered.pbf'),
                    extents=extents,
                    )

        shutil.copy(os.path.join(CONF_DIR, 'build-config.json'),
                    filtered_graph_folder)
        shutil.copy(os.path.join(CONF_DIR, 'router-config.json'),
                    filtered_graph_folder)

    if os.path.exists(os.path.join(filtered_graph_folder, 'graph.obj')):
        logging.warning(
            'A graph.obj file already exists and will be used. To rebuild the transport graph delete the graph.obj file.')
    else:
        prepare_graph(filtered_graph_folder)


if __name__ == '__main__':
    main()
