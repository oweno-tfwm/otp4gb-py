# -*- coding: utf-8 -*-
"""Functionality for requesting isochrones from OTP."""

#### IMPORTS ####
import dataclasses

# Standard imports
import datetime as dt
import pytz
import logging
from typing import Any
from urllib import parse

# Third party imports
import pydantic
import requests
from shapely import geometry

# Local imports
from otp4gb import routing

# CONSTANTS ####
LOG = logging.getLogger(__name__)
ISOCHRONE_API_ROUTE = "otp/traveltime/isochrone"
# WALK is not supported standalone arg - mix with another irrelevant mode
WALK_PAIR_MODE = "FERRY"


#### CLASSES ####
@dataclasses.dataclass
class IsochroneParameters:
    location: geometry.Point
    departure_time: dt.datetime
    cutoff: list[dt.timedelta]
    modes: list[routing.Mode]

    def parameters(self) -> dict[str, Any]:

        modes = [i.value for i in self.modes]

        for i in range(len(modes)):
            print(modes[i], type(modes[i]))
            if modes[i] == "WALK":
                modes[i] = ",".join(("WALK", WALK_PAIR_MODE))

        # OTP4GB - Assumed GB Timezone
        time_zone = pytz.timezone("Europe/London")
        aware_time = self.departure_time.astimezone(time_zone)
        print(modes)
        return dict(
            location=[str(i) for i in self.location.coords],
            time=aware_time, # TODO Should include time zone +00:00
            cutoff=[_format_cutoff(i) for i in self.cutoff],
            modes=modes, # TODO Cannot just be WALK should be WALK and TRANSIT
        )
    

class IsochroneResult(pydantic.BaseModel):
    ...


#### FUNCTIONS ####
def _format_cutoff(cutoff: dt.timedelta) -> str:
    seconds = round(cutoff.total_seconds())
    minutes = 0
    hours = 0

    if seconds > 60:
        minutes, seconds = divmod(seconds, 60)

    if minutes > 60:
        hours, minutes = divmod(minutes, 60)

    text = ""
    for name, value in (("H", hours), ("M", minutes), ("S", seconds)):
        if value > 0:
            text += f"{value}{name}"

    return text


def get_isochrone(server_url: str, parameters: IsochroneParameters) -> IsochroneResult:
    url = parse.urljoin(server_url, ISOCHRONE_API_ROUTE)

    error_message = []

    requester = routing.request(url, parameters.parameters())
    for response in requester:

        if response.status_code == requests.codes.OK:
            if response.text is None:
                error_message.append(f"Retry {response.retry}: {response.message}")
                continue
            result = IsochroneResult.parse_raw(response.text)
            return result
        
        error_message.append(f"Retry {response.retry}: {response.message}")


if __name__ == "__main__":
    departure_time = dt.datetime(2020, 1, 13, 8, 30)
    # TESTING -- Leave below
    # Sheff:  53.383331, -1.466666
    # Newcastle: 55.020825, -1.652973
    #params = IsochroneParameters(geometry.Point(53.383331, -1.466666), departure_time, [dt.timedelta(seconds=480), dt.timedelta(seconds=600)], [routing.Mode.WALK])
    params = IsochroneParameters(geometry.Point(53.383331, -1.466666),
                                 dt.datetime(2020, 1, 13, 8, 30),
                                 [dt.timedelta(seconds=120), dt.timedelta(seconds=240),
                                  dt.timedelta(seconds=360), dt.timedelta(seconds=480),
                                  dt.timedelta(seconds=600)],
                                 [routing.Mode.WALK])
    req = requests.Request("GET", parse.urljoin("http://localhost:8080", ISOCHRONE_API_ROUTE), params=params.parameters())
    prepared = req.prepare()

    # req url now incorrectly formatted. replace + with T to specify time
    prepared.url = prepared.url.replace(departure_time.strftime("%Y-%m-%d+%H"),
                                        departure_time.strftime("%Y-%m-%dT%H"))

    print(prepared.url)
