# -*- coding: utf-8 -*-
"""Module to send requests to the OTP routing API."""

##### IMPORTS #####
# Standard imports
import datetime
import logging
import re
from typing import Any, Optional, Union

# Third party imports
import pydantic
import requests
from shapely import geometry

# Local imports

##### CONSTANTS #####
LOG = logging.getLogger(__name__)
ROUTER_API_ROUTE = "otp/routers/default/plan"

##### CLASSES #####
class RoutePlanParameters(pydantic.BaseModel):
    fromPlace: geometry.Point
    toPlace: geometry.Point
    date: datetime.date
    time: datetime.time
    mode: Union[list[str], str]
    arriveBy: bool = False
    wheelchair: bool = False
    debugItineraryFilter: bool = False
    showIntermediateStops: bool = False
    maxWalkDistance: float = 1000
    locale: str = "en"

    class Config:
        arbitrary_types_allowed = True

    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {k: str(v) for k, v in self.dict().items()}

        # Update any parameters which require specific creation
        params.update(
            {
                "fromPlace": [str(i) for i in self.fromPlace.coords],
                "toPlace": [str(i) for i in self.toPlace.coords],
                "time": self.time.strftime("%H:%M%p"),
                "date": self.date.strftime("%m-%d-%Y"),  # date in USA format
                "mode": ",".join([s.upper() for s in self.mode]),
            }
        )
        return params

    @pydantic.validator("fromPlace", "toPlace", pre=True)
    def _validate_point(cls, value) -> geometry.Point:
        if isinstance(value, geometry.Point):
            return value

        if isinstance(value, str):
            match = re.match(
                r"^[(\s]*"  # Optional starting bracket
                r"([-+]?\d+(?:\.\d+)?)[,\s]+"  # 1st int or float with separator
                r"([-+]?\d+(?:\.\d+)?)"  # 2nd number
                r"(?:[,\s]+([-+]?\d+(?:\.\d+)?))?"  # optional 3rd number with separator
                r"[\s)]*$",  # optional ending bracket
                value.strip(),
            )
            if match is None:
                raise ValueError(f"invalid point value string '{value}'")

            value = [n for n in match.groups() if n is not None]

        if isinstance(value, (tuple, list)):
            if len(value) in (2, 3):
                return geometry.Point(*[float(x) for x in value])
            raise ValueError(f"point should have 2 coordinates not {len(value)}")

        raise TypeError(
            f"expected collection or string of coordinates for point not '{value}'"
        )

    @pydantic.validator("date", pre=True)
    def _valid_date(cls, value) -> datetime.date:
        if isinstance(value, datetime.date):
            return value

        if isinstance(value, str):
            return datetime.datetime.strptime(value.strip(), "%m-%d-%Y").date()

        raise TypeError(f"expected date or string not {type(value)}")

    @pydantic.validator("time", pre=True)
    def _valid_time(cls, value) -> datetime.date:
        if isinstance(value, datetime.time):
            return value

        if isinstance(value, str):
            return datetime.datetime.strptime(value.strip(), "%H:%M%p").time()

        raise TypeError(f"expected time or string not {type(value)}")


class Place(pydantic.BaseModel):
    name: str
    id: Optional[str] = None
    zone_system: Optional[str] = None
    lon: float
    lat: float

class Itinerary(pydantic.BaseModel):
    duration: int
    startTime: datetime.datetime
    endTime: datetime.datetime
    walkTime: int
    transitTime: int
    waitingTime: int
    walkDistance: float
    walkLimitExceeded: bool
    generalizedCost: int
    elevationLost: int
    elevationGained: int
    transfers: int
    legs: list
    tooSloped: bool
    # arrivedAtDestinationWithRentedBicycle: bool


class Plan(pydantic.BaseModel):
    date: datetime.datetime
    from_: Place = pydantic.Field(alias="from")
    to: Place
    itineraries: list[Itinerary]

    class Config:
        allow_population_by_field_name = True

class RoutePlanError(pydantic.BaseModel):
    id: int
    msg: str
    message: str
    missing: Optional[list[str]] = None

class RoutePlanResults(pydantic.BaseModel):
    requestParameters: RoutePlanParameters
    plan: Optional[Plan] = None
    error: Optional[RoutePlanError] = None
    debugOutput: Optional[dict] = None
    elevationMetadata: Optional[dict] = None


##### FUNCTIONS #####
def get_route_itineraries(
    server_url: str, parameters: RoutePlanParameters
) -> tuple[str, RoutePlanResults]:
    # TODO Use urllib.parse.urljoin
    if not server_url.endswith("/"):
        server_url += "/"
    url = server_url + ROUTER_API_ROUTE

    r = requests.get(url, params=parameters.params())

    return r.url, RoutePlanResults.parse_raw(r.text)


if __name__ == "__main__":

    parameters = RoutePlanParameters(
        fromPlace="(53.76819584019795, -1.602630615234375 )",
        toPlace="53.79821757312943, -1.498260498046875",
        date=datetime.date(2019, 9, 10),
        time=datetime.time(8, 30),
        mode=["BUS"],
    )
    parameters = RoutePlanParameters(
        fromPlace="53.639349,-1.5139415",
        toPlace="53.643807,-1.4693124",
        date=datetime.date(2019, 9, 10),
        time=datetime.time(8, 30),
        mode=["BUS", "WALK"],
    )

    res_url, result = get_route_itineraries("http://localhost:8080", parameters)
    print(res_url, result)
