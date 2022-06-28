import logging
import operator
import os
from otp4gb.geo import buffer_geometry, parse_to_geo
from otp4gb.net import api_call


logger = logging.getLogger(__name__)


def build_run_spec(name_key, modes, centroids, arrive_by, travel_time_max, travel_time_step, max_walk_distance, server):
    items = []
    for _, destination in centroids.iterrows():
        name = destination[name_key]
        location = [destination.geometry.y, destination.geometry.x]
        for mode in modes:
            cutoffs = [('cutoffSec', str(c*60))
                        for c in range(travel_time_step, travel_time_max+1, travel_time_step)]
            query = [
                ('fromPlace', ','.join([str(x) for x in location])),
                ('mode', mode),
                ('date', arrive_by.date()),
                ('time', arrive_by.time()),
                ('maxWalkDistance', str(max_walk_distance)),
                ('arriveby', 'false'),
            ] + cutoffs
            url = server.get_url('isochrone', query=query)
            batch_spec = {
                'name': name,
                'travel_time': arrive_by,
                'url': url,
                'mode': mode,
            }
            items.append(batch_spec)
    return items


def setup_worker(config):
    global output_dir, centroids, buffer_size, FILENAME_PATTERN
    output_dir = config.get('output_dir')
    centroids = config.get('centroids')
    buffer_size = config.get('buffer_size')
    FILENAME_PATTERN = config.get('FILENAME_PATTERN')


def run_batch(batch_args):
    # For each destination
    #   Calculate travel isochrone up to a number of cutoff times (1 to 90 mins)
    #   Write isochrone
    #   Calculate all possible origins within travel time by minutes
    #

    logger.debug('args = %s', batch_args)
    url, name, mode, travel_time = operator.itemgetter(
        'url', 'name', 'mode', 'travel_time')(batch_args)

    logger.info('Processing %s for %s', mode, name)

    logger.debug('Getting URL %s', url)
    data = api_call(url)

    data = parse_to_geo(data)

    data = buffer_geometry(data)

    for i in range(data.shape[0]):
        row = data.iloc[[i]]
        journey_time = row['time']
        geojson_file = os.path.join(
            output_dir,
            FILENAME_PATTERN.format(
                location_name=name,
                mode=mode,
                buffer_size=buffer_size,
                arrival_time=travel_time.isoformat(),
                journey_time=str(int(journey_time/60)).rjust(4, '_'),
            )
        )
        with open(geojson_file, 'w') as f:
            f.write(row.to_json())
