import georinex
import pytest
from pathlib import Path
import logging


def log_error_and_raise_value_exception(error_message):
    logging.error(error_message)
    raise ValueError(error_message)


def parse_rinex_3_obs(file: Path):
    if not file.is_absolute():
        log_error_and_raise_value_exception(f"The path to the file to be parsed ({file}) should be an absolute path "
                                            "such as /home/jan/prx_data/file.rnx as opposed to a relative path "
                                            "such as ../prx_data/file.rnx")
    header = georinex.rinexheader(file)
    if not header["version"] >= 3:
        log_error_and_raise_value_exception(
            f"Looks like the passed-in file ({file}) is not a RINEX 3.xx file, the version says {header['version']}"
        )
    parsed = georinex.load(file)
    if not parsed.attrs["time_system"] == "GPS":
        log_error_and_raise_value_exception(
            f"Expecting the passed-in file ({file}) to use GPST time, uses {parsed.attrs['time_system']}")
    return parsed
