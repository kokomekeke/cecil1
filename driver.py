from __future__ import annotations
from collections import deque
import logging
import multiprocessing.queues
import threading
import time
import traceback
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Dict, Optional, Set, Deque
from server.cecil.utils import clear_queue
import pynmea2
from Pyro5.server import expose
from server.cecil.config import get_config, SerialDeviceType
from server.cecil.algo.models import (
    SID,
    PSID,
    Antenna,
    AntennaPhase,
    RawMagnetometerReading,
    Position,
    Sat,
    SatHistory,
    Temperature,
)
from server.cecil.daq.parser import (
    parse_gga_sentence,
    parse_magnetometer_sentence,
    parse_phase_sentence,
    parse_temperature_sentence,
)
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from server.cecil.custom_typing import RawSentence

logger = logging.getLogger(__name__)


class Driver:
    def __init__(self, raw_queue: multiprocessing.queues.Queue[RawSentence]):
        self._do_reset = False
        self.antenna: Optional[Antenna] = None
        self.position: Optional[Position] = None
        self.magnetometer: Optional[RawMagnetometerReading] = None
        self.temprature: Optional[Temperature] = None
        self.status: Optional[str] = None
        self.error_status: Optional[str] = None

        self._cfg = get_config()
        self._encoding = self._cfg.device.ENCODING
        self._raw_queue = raw_queue
        self.satellites: Dict[str, SatHistory] = {}

        self.antenna_phase: AntennaPhase = AntennaPhase.A

        self.magnetometer_calibration_mode = False
        self.magnetometer_readings: Deque[RawMagnetometerReading] = deque(
            maxlen=self._cfg.magnetometer.AVG_SIZE
        )

        self.lock = threading.Lock()
        threading.Thread(target=self._loop).start()

    @expose
    def reset(self):
        print("RESET DUCK")
        self._do_reset = True

    def _check_for_reset(self):
        if self._do_reset:
            logger.warning("Starting driver reset")
            clear_queue(self._raw_queue)
            self.satellites.clear()
            self.antenna = None
            self.position = None
            self.magnetometer = None
            self.temprature = None
            self.status = None
            self.error_status = None
            logger.warning("Finished driver reset")
            self._do_reset = False

    def _loop(self):
        while True:
            with self.lock:
                self._check_for_reset()
                self._drop_old_satellites()

            sentence = self._raw_queue.get()
            decoded = sentence.raw.decode(self._encoding).strip()
            # print("raw: ", decoded)

            if len(decoded) == 0:
                logger.error("Received empty sentence")
                continue

            with self.lock:
                match decoded[0]:
                    case SID.PROPRIETARY:
                        # print("prop: ", decoded)
                        self._sort_proprietary_sentence(decoded)
                    case SID.STANDARD:
                        self._sort_regular_sentence(decoded)
                    case _:
                        logger.error(f"Unknown sentence: {decoded}")

    def _drop_old_satellites(self):
        to_drop: Set[int] = set()
        for sat in self.satellites.values():
            if sat.above_time_limit:
                logger.warning(
                    f"Dropping satellite {sat.prn}, above time limit ({self._cfg.daq.MAX_AGE_TIME} ms)"
                )
                to_drop.add(sat.ident)

        for ident in to_drop:
            self.satellites.pop(str(ident))

    def _process_gsv(self, sentence: pynmea2.GSV):
        """GSV sentences are split into multiple messages, each containing 4 satellites."""

        print("\n\nsentence: ", sentence)

        # Get the total number of messages
        total_number = sentence.num_messages
        sequence_number = sentence.msg_num
        sat_in_view = sentence.num_sv_in_view

        if not all([self.antenna, self.antenna_phase]):
            logger.warning("No antenna or antenna phase set")
            return

        logger.debug(
            f"{sequence_number}/{total_number} {sat_in_view} satellites in view"
        )
        for i in range(1, 5):
            print(i, ')')
            print("talker: ", sentence.talker)
            prn = getattr(sentence, f"sv_prn_num_{i}")
            prn = int(prn) if prn else None
            print("prn: ", prn)

            azimuth = getattr(sentence, f"azimuth_{i}")
            azimuth = int(azimuth) if azimuth else None
            print("Az: ", azimuth)

            elevation = getattr(sentence, f"elevation_deg_{i}")
            elevation = int(elevation) if elevation else None
            print("El: ", elevation)

            snr = getattr(sentence, f"snr_{i}")
            snr = int(snr) if snr else None
            print("snr: ", snr)

            # if snr is None or azimuth is None or elevation is None or prn is None:
            #     continue
            if azimuth is None or elevation is None or prn is None:
                continue

            # print("ANTENNA PHASE: ", self.antenna_phase)

            sat = Sat(
                sentence.talker,
                prn,
                azimuth,
                elevation,
                snr,
                self.antenna_phase,
                time.time(),
            )

            # print("ident: ", sat.ident)
            # print("talker: ", sat.talker)
            # nainnentol
            if str(sat.ident) not in self.satellites:
                # print("sat added")
                self.satellites[str(sat.ident)] = SatHistory(
                    talker=sat.talker,
                    prn=sat.prn,
                    avg_size=self._cfg.daq.AVG_SIZE,
                    max_age=self._cfg.daq.MAX_AGE,
                )
                # print("c: ", self.antenna.counter)
                print("satantenna: ", self.satellites[str(sat.ident)])

            if not self.antenna:
                logger.warning("Antenna counter not available")
                continue

            print("phase:", self.antenna.phase)
            print("antenna.counter: ", self.antenna.counter)
            age = self.satellites[str(sat.ident)].update(self.antenna.counter, sat)
            if self.satellites[str(sat.ident)].above_age_limit:
                logger.debug(f"Satellite {sat.ident} is too old, {age}")

    def _sort_proprietary_sentence(self, decoded: str):
        psid = decoded[:3]
        try:
            match psid:
                case PSID.STATUS:
                    self.status = decoded[4:]

                case PSID.ERROR_STATUS:
                    self.error_status = decoded[4:]

                case PSID.ANTENNA_PHASE:
                    # print("decoded: ", decoded)
                    # print("antennaPhase: ", decoded[4])
                    self.antenna_phase = AntennaPhase(decoded[4])

                    self.antenna = parse_phase_sentence(decoded)
                    print("antenna: ", self.antenna)
                    logger.debug(f"Phase sentence: {self.antenna}")
                case PSID.TEMPERATURE:
                    self.temprature = parse_temperature_sentence(decoded)

                case PSID.MAGNETOMETER:
                    self.magnetometer = parse_magnetometer_sentence(
                        decoded,
                        self._cfg.magnetometer.FLIP_MAGNETOMETER_X,
                        self._cfg.magnetometer.FLIP_MAGNETOMETER_Y,
                        self._cfg.magnetometer.FLIP_MAGNETOMETER_Z,
                    )
                    self.magnetometer_readings.append(self.magnetometer)
                case _:
                    logger.warning(f"Got not impl proprietary sentence: {decoded}")
        except Exception as e:
            logger.error(
                f"{psid}: Failed to parse proprietary sentence: {decoded}, {e}"
            )

    def _sort_regular_sentence(self, decoded: str):
        try:
            sentence = pynmea2.parse(decoded)
        except pynmea2.ChecksumError as e:
            logger.error(f"Checksum error: {decoded}")
            traceback.print_exc()
            return
        except pynmea2.ParseError as e:
            logger.error(f"Unable to parse with pynmea: {decoded}")
            traceback.print_exc()
            return
        except Exception as e:
            logger.error(f"Unable to parse with pynmea: {decoded}")
            traceback.print_exc()
            return

        try:
            match sentence.sentence_type:
                case SID.GSV:
                    self._process_gsv(sentence)
                case SID.GGA:
                    # MOCK interface esetén, hogy ne mozogjon a pozíció
                    if get_config().device.TYPE == SerialDeviceType.MOCK and not get_config().mocking.DYNAMIC_SIM:
                        self.position = parse_gga_sentence(pynmea2.parse("$GPGGA,213512.10,4729.8747,N,01903.1441,E,1,14,3.1,-13.0,M,-45.3,M,,*53"))
                        print("position: ", self.position)
                    else:
                        self.position = parse_gga_sentence(sentence)
        except Exception as e:
            logger.error(f"Failed to process sentence: {decoded}, {e}")
            traceback.print_exc()
