import pytest
import helpers
import constants
import pandas as pd


def test_rinex_header_time_string_2_timestamp_ns():
    assert (
        helpers.timedelta_2_nanoseconds(helpers.timestamp_2_timedelta(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     6     0     0    0.0000000     GPS"
            ),
            "GPST",
        ))
        == 0
    )
    assert (
        helpers.timedelta_2_nanoseconds(helpers.timestamp_2_timedelta(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     6     0     0    1.0000000     GPS"
            ),
            "GPST",
        ))
        == constants.cNanoSecondsPerSecond
    )
    timestamp = helpers.rinex_header_time_string_2_timestamp_ns(
        "  1980     1     6     0     0    1.0000001     GPS"
    )
    timedelta = helpers.timestamp_2_timedelta(timestamp, "GPST")

    assert helpers.timedelta_2_nanoseconds(timedelta) == constants.cNanoSecondsPerSecond + 100
    assert (
        helpers.timedelta_2_nanoseconds(helpers.timestamp_2_timedelta(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     7     0     0    0.0000000     GPS"
            ),
            "GPST",
        ))
        == constants.cSecondsPerDay * constants.cNanoSecondsPerSecond
    )
