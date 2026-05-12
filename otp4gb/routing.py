# -*- coding: utf-8 -*-
"""Module to send requests to the OTP routing API."""

##### IMPORTS #####
from __future__ import annotations
import dataclasses

# Standard imports
import datetime
import enum
import logging
import re
from typing import Any, Optional, Union
from urllib import parse

# Third party imports
import pydantic
import requests
from shapely import geometry

# Local imports

##### CONSTANTS #####
LOG = logging.getLogger(__name__)
ROUTER_API_ROUTE = "otp/routers/default/plan"
REQUEST_TIMEOUT = 200
REQUEST_RETRIES = 5
OTP_ERRORS = {
    "NO_TRANSIT": 406,
    "TOO_CLOSE": 409,
    "TRIP_IMPOSSIBLE": 400,
    "GEOCODE_TO_NOT_FOUND": 450,
}


##### CLASSES #####
# see otp source TransitMode.modesConsideredTransitByUsers()  and enum ApiRequestMode  and  enum Qualifier
#
# this is far more complicated than it should be. The OTP documentation uses internal names, and the values to use in the APIs are not clearly listed.
# the rest API is being depreciated, which makes the documentation even less helpful.
# it has also been extended, so it seems that the mode string supplied to the API can be a base mode with a series of underscore delimited qualifiers tacked on the end.
# in v 2.5 of otp the qualifiers are :-
# RENT,PARK,PICKUP,DROPOFF,ACCESS,EGRESS,DIRECT,HAIL
# 
# the base (ApiRequestMode) modes are :-
# WALK,BICYCLE,SCOOTER,CAR,TRAM,SUBWAY,RAIL,BUS,FERRY,CABLE_CAR,GONDOLA,FUNICULAR,TRANSIT,AIRPLANE,TROLLEYBUS,MONORAIL,CARPOOL,TAXI,FLEX
#
# Transit = RAIL,COACH,SUBWAY,BUS,TRAM,FERRY,AIRPLANE,CABLE_CAR,GONDOLA,FUNICULAR,TROLLEYBUS,MONORAIL,TAXI
# (note how this includes taxi - basically everything apart from carpool) 
#
# see also this code in the OTP web client.
'''otp.config.modes = {
    "TRANSIT,WALK"             : _tr("Transit"),
    "BUS,WALK"                 : _tr("Bus Only"),
    "TRAM,RAIL,SUBWAY,FUNICULAR,GONDOLA,WALK": _tr("Rail Only"),
    "AIRPLANE,WALK"            : _tr("Airplane Only"),
    "BUS,TRAM,RAIL,FERRY,SUBWAY,FUNICULAR,GONDOLA,WALK" : _tr("Transit, No Airplane"),
    "BICYCLE"                  : _tr('Bicycle Only'),
    "TRANSIT,BICYCLE"          : _tr("Bicycle &amp; Transit"),
    "WALK"                     : _tr('Walk Only'),
    "CAR"                      : _tr('Car Only'),
    "CAR_PICKUP"               : _tr('Taxi'),
    "CAR_PARK,TRANSIT"         : _tr('Park and Ride'),
    "CAR_PICKUP,TRANSIT"       : _tr('Ride and Kiss (Car Pickup)'),
    "CAR_DROPOFF,TRANSIT"      : _tr('Kiss and Ride (Car Dropoff)'),
    "BICYCLE_PARK,TRANSIT"     : _tr('Bike and Ride'),
    //uncomment only if bike rental exists in a map
    // TODO: remove this hack, and provide code that allows the mode array to be configured with different transit modes.
    //       (note that we've been broken for awhile here, since many agencies don't have a 'Train' mode either...this needs attention)
    // IDEA: maybe we start with a big array (like below), and the pull out modes from this array when turning off various modes...
    'BICYCLE_RENT'             : _tr('Rented Bicycle'),
    'TRANSIT,BICYCLE_RENT'     : _tr('Transit & Rented Bicycle'),
    'SCOOTER_RENT'             : _tr('Rented Scooter'),
    'TRANSIT,SCOOTER_RENT'     : _tr('Transit & Rented Scooter'),
    "FLEX_ACCESS,WALK,TRANSIT" : _tr('Transit with flex access'),
    "FLEX_EGRESS,WALK,TRANSIT" : _tr('Transit with flex egress'),
    "FLEX_ACCESS,FLEX_EGRESS,TRANSIT" : _tr('Transit with flex access and egress'),
    "FLEX_DIRECT"              : _tr('Direct flex search'),
    "CARPOOL,WALK"             : _tr("Carpool"),
    "CAR_HAIL,TRANSIT,WALK"    : _tr("Car hailing and transit")
};'''


class Mode(enum.StrEnum):
    TRANSIT = "TRANSIT" #RAIL, COACH, SUBWAY, BUS, TRAM, FERRY, AIRPLANE, CABLE_CAR, GONDOLA, FUNICULAR, TROLLEYBUS, MONORAIL, TAXI;
    BUS = "BUS" #included in 'transit'
    RAIL = "RAIL" #included in 'transit'
    TRAM = "TRAM" #included in 'transit'
    WALK = "WALK"
    BICYCLE = "BICYCLE"
    BICYCLE_PARK = "BICYCLE_PARK" #just adding in the _park qualifier for bicycle at the moment
    SCOOTER = "SCOOTER"
    CAR = "CAR"
    SUBWAY = "SUBWAY" #included in 'transit'
    FERRY = "FERRY" #included in 'transit'
    CABLE_CAR = "CABLE_CAR" #included in 'transit'
    GONDOLA = "GONDOLA" #included in 'transit'
    FUNICULAR = "FUNICULAR" #included in 'transit'
    AIRPLANE = "AIRPLANE" #included in 'transit'
    TROLLEYBUS = "TROLLEYBUS" #included in 'transit'
    MONORAIL = "MONORAIL" #included in 'transit'
    CARPOOL = "CARPOOL"
    TAXI = "TAXI" #included in 'transit'
    FLEX = "FLEX"

    @staticmethod
    def transit_modes() -> set[Mode]:
        return {Mode.TRANSIT, Mode.BUS, Mode.RAIL, 
                Mode.TRAM, Mode.SUBWAY, Mode.FERRY, 
                Mode.CABLE_CAR, Mode.GONDOLA, Mode.FUNICULAR, 
                Mode.AIRPLANE, Mode.TROLLEYBUS, Mode.MONORAIL, Mode.TAXI }


class RoutePlanParameters(pydantic.BaseModel):
    fromPlace: geometry.Point
    toPlace: geometry.Point
    date: datetime.date
    time: datetime.time
    mode: Union[list[str], str]
    arriveBy: bool = False
    searchWindow: Optional[int] = None
    wheelchair: bool = False
    debugItineraryFilter: bool = False
    showIntermediateStops: bool = False
    maxWalkDistance: float = 1000
    locale: str = "en"

    class Config:
        arbitrary_types_allowed = True

    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            k: str(v) for k, v in self.dict().items() if v is not None
        }

        # Update any parameters which require specific creation
        params.update(
            {
                "fromPlace": [str(i) for i in self.fromPlace.coords],
                "toPlace": [str(i) for i in self.toPlace.coords],
                "time": self.time.strftime("%I:%M%p"),
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
            f"expected collection or string of coordinates for point not '{type(value)}'"
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


class Leg(pydantic.BaseModel):
    startTime: datetime.datetime
    endTime: datetime.datetime
    distance: float
    mode: Mode
    transitLeg: bool
    duration: int

    class Config:
        extra = pydantic.Extra.allow


class Itinerary(pydantic.BaseModel):
    duration: int
    startTime: datetime.datetime
    endTime: datetime.datetime
    walkTime: int
    transitTime: int
    waitingTime: int
    walkDistance: float
    walkLimitExceeded: bool
    otp_generalised_cost: int = pydantic.Field(alias="generalizedCost")
    elevationLost: int
    elevationGained: int
    transfers: int
    legs: list[Leg]
    tooSloped: bool
    generalised_cost: Optional[float] = None
    # arrivedAtDestinationWithRentedBicycle: bool

    class Config:
        allow_population_by_field_name = True


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


@dataclasses.dataclass
class _FakeResponse:
    """Storing data when request errors, within `get_route_itineraries`."""

    url: str
    status_code: int
    reason: str


##### FUNCTIONS #####
def get_route_itineraries(
    server_url: str, parameters: RoutePlanParameters
) -> tuple[str, RoutePlanResults]:
    """Request routing from Open Trip Planner server.

    Parameters
    ----------
    server_url : str
        URL for the OTP server.
    parameters : RoutePlanParameters
        Parameters for the route to request.

    Returns
    -------
    str
        Request URL, with parameters.
    RoutePlanResults
        OTP routing response.
    """

    def add_error(message: str) -> None:
        error_message.append(f"Retry {retries}: {message}")

    url = parse.urljoin(server_url, ROUTER_API_ROUTE)

    retries = 0
    error_message = []
    end = False
    while True:
        req = requests.Request("GET", url, params=parameters.params())
        prepared = req.prepare()

        try:
            session = requests.Session()
            response = session.send(prepared, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as error:
            msg = f"{error.__class__.__name__}: {error}"
            add_error(msg)
            response = _FakeResponse(url=prepared.url, status_code=-10, reason=msg)

        if response.status_code == requests.codes.OK:
            result = RoutePlanResults.parse_raw(response.text)
            if result.error is None:
                return response.url, result
            if result.error.id in OTP_ERRORS.values():
                return response.url, result

            add_error(
                f"OTP Error {result.error.id}: {result.error.msg} {result.error.message}"
            )

        if end or retries > REQUEST_RETRIES:
            error_message.append("max retries reached")
            result = RoutePlanResults(
                requestParameters=parameters,
                error=RoutePlanError(
                    id=response.status_code,
                    msg=f"Response {response.status_code}: {response.reason}",
                    message="\n".join(error_message),
                ),
            )
            return response.url, result

        retries += 1
