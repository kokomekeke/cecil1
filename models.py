import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, List, Literal, Optional

import cardinality

from server.cecil.config import get_config
from server.cecil.generics import CaporDict

logger = logging.getLogger(__name__)


class SID(StrEnum):
    STANDARD = "$"
    GSV = "GSV"
    GGA = "GGA"
    PROPRIETARY = "#"


class PSID(StrEnum):
    """Proprietary Sentence ID's"""

    STATUS = "#00"
    ERROR_STATUS = "#01"
    ANTENNA_PHASE = "#02"
    TEMPERATURE = "#11"
    MAGNETOMETER = "#10"


class AntennaPhase(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Talker(StrEnum):
    GPS = "GP"
    GLONASS = "GL"
    GALIEO = "GA"
    BEIDOU = "GB"
    GNSS = "GN"
    QZSS = "GQ"


def talker_to_offset(talker: Talker) -> int | None:
    return {
        Talker.GALIEO: 1000,
        Talker.BEIDOU: 2000,
        Talker.GLONASS: 3000,
        Talker.GNSS: 4000,
        Talker.GPS: 5000,
        Talker.QZSS: 6000,
    }.get(talker, None)


@dataclass(frozen=True)
class Antenna:
    counter: int
    """Number of times antenna phase has changed"""
    phase: AntennaPhase
    """Current antenna phase"""
    interval: int
    """Interval in seconds between antenna phase changes"""
    offset: int
    """Offset in seconds between antenna phase changes"""
    last_change: float
    """Last time antenna phase changed"""


@dataclass(frozen=True)
class Sat:
    """A satellite in view of the GPS receiver"""

    talker: Talker
    prn: Optional[int]
    azimuth: Optional[int]
    elevation: Optional[int]
    SNR: Optional[int]
    phase: AntennaPhase
    timestamp: float

    @property
    def ident(self) -> int:
        offset = talker_to_offset(self.talker)
        if offset and self.prn:
            return offset + self.prn
        else:
            return -1000

    @property
    def is_valid(self) -> bool:
        """when satellite goes missing (SNR == 0) then invalidate satellite"""
        return self.SNR is not None and self.SNR > 0
        # return self.SNR is None


@dataclass(frozen=True)
class Temperature:
    internal: float
    external: float

    def __str__(self):
        return f"({self.internal}, {self.external})"


@dataclass(frozen=True)
class RawMagnetometerReading:
    x: float
    y: float
    z: float

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"


@dataclass(frozen=True)
class Position:
    """Global Positioning System Fix Data

    parsed from GGA sentence
    eg.: $GPGGA,115739.00,4158.8441367,N,09147.4416929,W,4,13,0.9,255.747,M,-32.00,M,01,0000*6E

    115739.00 = Time the position was recorded, in this case 11:57:39.00 UTC. This is always 6 digits, with an optional decimal place and one or two digits after the decimal for receivers with outputs faster than 1 Hz. The output has two digits for hours, two digits for minutes, and two digits for seconds. It can range from 000000 to 235959. The time is always in UTC, regardless of which time zone you’re in.

    4158.8441367 = Latitude in Degrees Minutes. This is a minimum of 4 digits, a decimal, and 2 more digits, but has the option to have additional digits after the decimal place. Of the 4 digits before the decimal, the first two are Degrees, the next two are Minutes. For this example, 41° 58.8441367 Minutes.

    N = North or South Hemisphere. This is always either ‘N’ or ‘S’. When converting to decimal degrees, the southern hemisphere is a negative number.

    09147.4416929 = Longitude in Degrees Minutes. This is a minimum of 5 digits, a decimal, and 2 more digits, but has the option to have additional digits after the decimal place. Of the 5 digits before the decimal, the first three are Degrees, the next two are Minutes. For this example, 91° 47.4416929 Minutes.

    W = East or West Hemisphere. This is always either ‘E’ or ‘W’. When converting to decimal degrees, the western hemisphere is a negative number.

    4 = Fix type. This is always a single number. Reportable solutions include:
        0 = Invalid, no position available.
        1 = Autonomous GPS fix, no correction data used.
        2 = DGPS fix, using a local DGPS base station or correction service such as WAAS or EGNOS.
        3 = PPS fix, I’ve never seen this used.
        4 = RTK fix, high accuracy Real Time Kinematic.
        5 = RTK Float, better than DGPS, but not quite RTK.
        6 = Estimated fix (dead reckoning).
        7 = Manual input mode.
        8 = Simulation mode.
        9 = WAAS fix (not NMEA standard, but NovAtel receivers report this instead of a 2).

    13 = Number of satellites in use. For GPS-only receivers, this will usually range from 4-12. For receivers that support additional constellations such as GLONASS, this number can be 21 or more.

    0.9 = Horizontal Dilution of Precision (HDOP). This is a unitless number indicating how accurate the horizontal position is. Lower is better.

    255.747 = Elevation or Altitude in Meters above mean sea level.

    M = This is always ‘M’, presumably for meters, the unit of the previous field.

    -32.00 = Height of the Geoid (mean sea level) above the Ellipsoid, in Meters.

    M = This is always ‘M’, presumably for meters, the unit of the previous field.

    01 = Age of correction data for DGPS and RTK solutions, in Seconds. Some receivers output this rounded to the nearest second, some will also include tenths of a second.

    0000 = Correction station ID number, used to help you determine which base station you’re using. This is almost always 4 digits.
    """

    talker: Talker
    timestamp: float
    latitude: float
    longitude: float
    fix_quality: int
    num_satellites: int
    horizontal_dilution: float
    altitude: float
    geoidal_separation: float
    age: float
    station_id: int

    @property
    def is_valid(self) -> bool:
        return self.fix_quality > 0


@dataclass
class Cycle:
    """A record of a satellite's history within a single cycle"""

    counter: int
    A: List[int] = field(default_factory=list)
    B: List[int] = field(default_factory=list)
    C: List[int] = field(default_factory=list)
    D: List[int] = field(default_factory=list)

    def __getitem__(
            self,
            phase: AntennaPhase
                   | str
                   | Literal["A"]
                   | Literal["B"]
                   | Literal["C"]
                   | Literal["D"],
    ) -> List[int]:
        return getattr(self, phase)

    @property
    def is_invalid(self) -> bool:
        return not self.is_valid

    @property
    def is_valid(self) -> bool:
        """Has at least one satellite in each phase"""
        return all(len(self[phase]) > 0 for phase in AntennaPhase)


@dataclass(frozen=True)
class FrozenCycle:
    """A record of a satellite's history within a single cycle"""

    counter: int
    A: List[int]
    B: List[int]
    C: List[int]
    D: List[int]
    avg_A: float
    avg_B: float
    avg_C: float
    avg_D: float
    azimuth: float

    def __getitem__(
            self,
            phase: AntennaPhase
                   | str
                   | Literal["A"]
                   | Literal["B"]
                   | Literal["C"]
                   | Literal["D"],
    ) -> List[int]:
        return getattr(self, phase)


class SatHistory:
    """A class to maintain a history of satellite data for a specific satellite.

    The `SatHistory` class is used to track and manage data for a particular satellite over time.
    It keeps a history of satellite signal strength (SNR) values for different antenna phases
    (A, B, C, D) within a single cycle. The class provides methods for updating the data,
    calculating averages, and determining the validity of the satellite data.

    Attributes:
        talker (Talker): The talker identifier associated with the satellite.
        prn (int): The pseudo-random noise (PRN) number identifying the satellite.
        avg_size (int): The number of samples to average over for each cycle.
        max_age (int): The maximum number of GSV messages to wait before invalidating a satellite.
        active_phase (AntennaPhase): The current active antenna phase (A, B, C, D).
        _active_cycle (Optional[ActiveCycle]): The active cycle of satellite data.
        _records (CaporDict[int, Cycle]): A dictionary to store historical cycles of satellite data.
        age (int): The age of the satellite data in GSV messages.
        valid (bool): Indicates whether the satellite data is valid.
        last_update (Optional[float]): The timestamp of the last update for the satellite data.
        latest_sat (Optional[Sat]): The latest information about the satellite.

    Methods:
        is_valid: Check if the satellite data is valid.
        above_age_limit: Check if the age of the satellite data exceeds the maximum age limit.
        above_time_limit: Check if the time since the last update exceeds a maximum time limit.
        ident: Calculate a unique identifier for the satellite based on talker and PRN.
        records: Get an iterable of all historical satellite data cycles.
        update(cycle: int, sat: Sat) -> int: Update the satellite data with new information.

    """

    def __init__(
            self, talker: Talker, prn: Optional[int], avg_size: int = 50, max_age: int = 60
    ):
        self.talker: Talker = talker
        self.prn: Optional[int] = prn

        self.avg_size: int = avg_size
        print("AVG SIZE: ", avg_size)
        """Number of samples to average over"""
        self.max_age: int = max_age
        """Number of GSV messages to wait before invalidating a satellite"""

        self.active_phase: AntennaPhase = AntennaPhase.A
        self.active_cycle: Optional[Cycle] = None
        self._records: CaporDict[int, FrozenCycle] = CaporDict(avg_size)

        self.age: int = 0
        self.valid: bool = True
        self.last_update: Optional[float] = None
        self.latest_sat: Optional[Sat] = None

    def __repr__(self) -> str:
        return f"SatHistory({self.talker}, {self.prn})"

    def __iter__(self) -> Iterable[FrozenCycle]:
        return self._records.values()

    def __contains__(self, key: int) -> bool:
        return key in self._records

    def __getitem__(self, key: int) -> FrozenCycle:
        return self._records[key]

    def __len__(self) -> int:
        return len(self._records)

    @property
    def is_valid(self) -> bool:
        return self.prn is not None

    @property
    def above_age_limit(self) -> bool:
        return self.age >= self.max_age

    @property
    def above_time_limit(self) -> bool:
        cfg = get_config()
        if self.last_update is None:
            return False

        max_age_ms = cfg.daq.MAX_AGE_TIME / 1000

        return self.last_update <= time.time() - max_age_ms

    @property
    def ident(self) -> int:
        offset = talker_to_offset(self.talker)
        if offset and self.prn:
            return offset + self.prn
        return -1000

    @property
    def records(self) -> Iterable[FrozenCycle]:
        return self._records.values()

    # megoldani, hogyha None egy snr, akkor az hozzátartozó többi fázis is érvénytelen legyen
    def update(self, cycle: int, sat: Sat) -> int:
        """Update satellite with new data"""

        # print("cycle: ", cycle)
        # print("sat: ", sat)
        if not sat.SNR:
            logger.debug(f"Satellite {sat.ident} went missing")
            self.valid = False
            self.active_phase = sat.phase
            self.age += 1
            return self.age

        if self.above_age_limit:
            logger.warning(f"SatHistory above age limit {sat.ident}, {self.age}")
            self._records.clear()

        # ez igy jo?
        if not self.valid:
            self.age += 1
            return self.age

        # Create the active cycle if it doesn't exist
        if self.active_cycle is None:
            self.active_cycle = Cycle(cycle)

        if cycle < self.active_cycle.counter:
            logger.warning(f"Received an earlier cycle {cycle}, ignoring.")
            return self.age

        if cycle in self._records:
            logger.warning(f"Cycle {cycle} already exists in records, ignoring.")
            return self.age

        # If the cycle has changed, add the active cycle to the history
        print(self.active_cycle.__str__())
        if cycle != self.active_cycle.counter:
            if self.active_cycle.is_valid:
                print("freeze")
                self._records[self.active_cycle.counter] = freeze_cycle(
                    self.active_cycle
                )
                # self.active_cycle = Cycle(cycle)
            else:
                logger.debug(
                    f"Active cycle is invalid {self.active_cycle.counter}, so not computing"
                )

                # Create a new active cycle
            self.active_cycle = Cycle(cycle)

        print("snr: ", sat.SNR)
        self.active_cycle[sat.phase].append(sat.SNR)
        print("active cycle: ", self.active_cycle[sat.phase].__str__())
        self.latest_sat = sat
        self.valid = True
        self.active_phase = sat.phase
        self.age = 0
        self.last_update = time.time()  # sat.timestamp
        return self.age


# az összes adatot átlagolja minden phaseben, vagy csak a valid phasek összetartozó adatait?
def freeze_cycle(active: Cycle) -> FrozenCycle:
    """Archive the active cycle"""
    if active.is_invalid:
        raise ValueError("Cannot freeze invalid cycle")

    cfg = get_config()

    avg_a = statistics.fmean(active.A)
    avg_b = statistics.fmean(active.B)
    avg_c = statistics.fmean(active.C)
    avg_d = statistics.fmean(active.D)

    print("avgs: ", avg_a, avg_b, avg_c, avg_d)

    azimuth = math.atan2(avg_a - avg_c, avg_b - avg_d)

    # apply az offset
    azimuth = (
            math.degrees(2 * math.pi + azimuth if azimuth < 0 else azimuth)
            + cfg.daq.AZ_OFFSET
    )

    return FrozenCycle(
        active.counter,
        active.A,
        active.B,
        active.C,
        active.D,
        avg_a,
        avg_b,
        avg_c,
        avg_d,
        azimuth,
    )
