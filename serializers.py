from __future__ import annotations
from datetime import time
import logging
import traceback
from typing import Any, Dict, Optional, Tuple, TypedDict, TYPE_CHECKING
import uuid

import orjson
from server.cecil.algo.functions import (
    calculate_avg_snr,
    calculate_azimuth_avg,
    calculate_snr_avg_azimuth,
    get_azimuths,
    get_latest_snr,
)
from server.cecil.algo.models import AntennaPhase, Sat, SatHistory

if TYPE_CHECKING:
    from server.cecil.detector.typing import Detection


logger = logging.getLogger(__name__)


class SerializedSatHistory(TypedDict):
    talker: str
    prn: Optional[int]
    ident: int
    calculated_azimuth: Optional[float]
    calculated_azimuth_az_avg: Optional[float]
    reported_azimuth: Optional[float]
    reported_elevation: Optional[float]
    SNR: Optional[float]
    avg_SNR_A: Optional[float]
    avg_SNR_B: Optional[float]
    avg_SNR_C: Optional[float]
    avg_SNR_D: Optional[float]
    latest_SNR_A: Optional[float]
    latest_SNR_B: Optional[float]
    latest_SNR_C: Optional[float]
    latest_SNR_D: Optional[float]
    is_valid: Optional[bool]
    age: Optional[int]
    last_update: Optional[float]


def serialize_sat_histories(
    histories: Dict[str, SatHistory]
) -> Tuple[uuid.UUID, Dict[str, SerializedSatHistory]]:
    update = {}
    for key, sat in histories.items():
        try:
            update[key] = serialize_sat_history(sat)
        except Exception as e:
            logger.error(f"Failed to serialize sat history for {key}: {e}")
            update.pop(key, None)
            traceback.print_exc()
    return uuid.uuid4(), update


def serialize_sat_history(history: SatHistory) -> SerializedSatHistory:
    # Left side algo
    avg_snr_a, avg_snr_b, avg_snr_c, avg_snr_d = calculate_avg_snr(history)
    algo_l_AZ = calculate_snr_avg_azimuth(avg_snr_a, avg_snr_b, avg_snr_c, avg_snr_d)


    # Right side algo
    azimuths = get_azimuths(history)
    algo_r_AZ = calculate_azimuth_avg(azimuths)

    # latest readings from any phase
    # latest_sat_azimuth = history.latest_sat.azimuth if history.latest_sat else None
    # latest_sat_elevation = history.latest_sat.elevation if history.latest_sat else None
    latest_sat_snr = history.latest_sat.SNR if history.latest_sat else None

    return {
        "talker": history.talker,
        "prn": history.prn,
        "ident": history.ident,
        "calculated_azimuth": algo_l_AZ,
        "calculated_azimuth_az_avg": algo_r_AZ,
        "reported_azimuth": history.latest_sat.azimuth if history.latest_sat else None,
        "reported_elevation": history.latest_sat.elevation
        if history.latest_sat
        else None,
        "SNR": latest_sat_snr,
        "avg_SNR_A": avg_snr_a,
        "avg_SNR_B": avg_snr_b,
        "avg_SNR_C": avg_snr_c,
        "avg_SNR_D": avg_snr_d,
        "latest_SNR_A": get_latest_snr(history, AntennaPhase.A),
        "latest_SNR_B": get_latest_snr(history, AntennaPhase.B),
        "latest_SNR_C": get_latest_snr(history, AntennaPhase.C),
        "latest_SNR_D": get_latest_snr(history, AntennaPhase.D),
        "is_valid": history.is_valid,
        "age": history.age,
        "last_update": history.last_update,
    }


def serialize_detection(detection: Detection, digits: int = 4) -> Dict[str, Any]:
    return {
        "identifier": str(detection.identifier),
        "timestamp": detection.timestamp.isoformat(),
        "suspicious_sat_count": detection.suspicious_sat_count,
        "abs_spoofer_angle": round(detection.abs_spoofer_angle, digits),
        "rel_spoofer_angle": round(detection.rel_spoofer_angle, digits),
        "confidence": round(detection.confidence, digits),
        "mean": round(detection.mean, digits),
        "variance": round(detection.variance, digits),
        "std_dev": round(detection.std_dev, digits),
        "alert_level": detection.alert_level,
    }


def default(obj: Any) -> Any:
    if isinstance(obj, SatHistory):
        return serialize_sat_history(obj)
    if isinstance(obj, Sat):
        return obj.asdict()
    if isinstance(obj, time):
        return obj.isoformat()

    raise TypeError


def jsonify(obj: Any) -> bytes:
    return orjson.dumps(obj, default=default)
