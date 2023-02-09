import hashlib
from pathlib import Path
import logging
from datetime import datetime
import math
import constants
import numpy as np

logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.DEBUG,
)


def get_logger(label):
    return logging.getLogger(__name__)


# From https://stackoverflow.com/a/3431838
def md5(file: Path):
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def rinex_header_time_string_2_gpst_ns(time_string: str) -> int:
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
    date = np.datetime64(datetime64_string)
    delta = date - constants.cGpstEpoch
    return delta.astype("timedelta64[ns]").astype(np.int64)
