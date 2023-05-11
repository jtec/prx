import hashlib
from pathlib import Path
import logging
import constants
import numpy as np
import pandas as pd
import glob
import subprocess
import math

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


def get_logger(label):
    return logging.getLogger(label)


def prx_root() -> Path:
    return Path(__file__).parent.parent


# From https://stackoverflow.com/a/3431838
def md5_of_file_content(file: Path):
    assert file.exists(), f"Looks like {file} does not exist."
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def timestamp_2_timedelta(timestamp: pd.Timestamp, time_scale):
    assert type(timestamp) == pd.Timestamp, "timestamp must be of type pd.Timestamp"
    # RINEX 3 adds the offset between GPST and GST/QZSST/IRNSST epochs, so we can use the GPST epoch here.
    # The SBAST epoch is the same as the GPST epoch.
    if time_scale == "GPST" or time_scale == "SBAST" or time_scale == "QZSST" or time_scale == "IRNSST" or time_scale == "GST":
        return timestamp - constants.cGpstUtcEpoch
    if time_scale == "BDT":
        # TODO double-check the offset between BDT and GPST
        return (timestamp - constants.cGpstUtcEpoch) - \
            pd.Timedelta(1356*constants.cSecondsPerWeek, "seconds") \
            - pd.Timedelta(14, "seconds")
    # The GLONASS epoch is (probably) the UTC epoch, to keep the Timedelta within the same order of magnitude
    # as for the other constellations, we use an arbitrary epoch here.
    if time_scale == "GLONASST":
        return timestamp - constants.cArbitraryGlonassUtcEpoch
    assert False, f"Time scale {time_scale} not supported."

def timedelta_2_weeks_and_seconds(time_delta: pd.Timedelta):
    assert type(time_delta) == pd.Timedelta, "time_delta must be of type pd.Timedelta"
    weeks = math.floor(time_delta.delta/constants.cNanoSecondsPerWeek)
    week_nanoseconds = time_delta.delta - weeks * constants.cNanoSecondsPerWeek
    return weeks, np.float64(week_nanoseconds) / constants.cNanoSecondsPerSecond

def timedelta_2_seconds(time_delta: pd.Timedelta):
    assert type(time_delta) == pd.Timedelta, "time_delta must be of type pd.Timedelta"
    integer_seconds = np.float64(round(time_delta.total_seconds()))
    fractional_seconds = np.float64(time_delta.delta - integer_seconds * constants.cNanoSecondsPerSecond)
    return integer_seconds + fractional_seconds


def rinex_header_time_string_2_timestamp_ns(time_string: str) -> pd.Timestamp:
    elements = time_string.split()
    time_scale = elements[-1]
    assert time_scale == "GPS", "Time scales other than GPST not supported yet"
    year = int(elements[0])
    month = int(elements[1])
    day = int(elements[2])
    hour = int(elements[3])
    minute = int(elements[4])
    second = float(elements[5])
    datetime64_string = (
        f"{year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:012.9f}"
    )
    timestamp = pd.Timestamp(np.datetime64(datetime64_string))
    return timestamp


def repair_with_gfzrnx(file):
    gfzrnx_binaries = glob.glob(
        str(prx_root().joinpath("tools/gfzrnx/**gfzrnx**")), recursive=True
    )
    for gfzrnx_binary in gfzrnx_binaries:
        command = f" {gfzrnx_binary} -finp {file} -fout {file}  -chk -kv -f"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            log.info(f"Ran gfzrnx file repair on {file}")
            return file
    assert False, "gdzrnx file repair run failed!"


def deg_2_rad(angle_deg):
    return angle_deg * np.pi / 180
