import pytest
import helpers
import constants
import pandas as pd


def test_timestamp_2_gpst_n():
    assert (
        helpers.timestamp_2_gpst_ns(pd.Timestamp("1980-01-07T00:00:00.000000000"))
        == constants.cSecondsPerDay * constants.cNanoSecondsPerSecond
    )


def test_rinex_header_time_string_2_datetime64():
    assert (
        helpers.timestamp_2_gpst_ns(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     6     0     0    0.0000000     GPS"
            )
        )
        == 0
    )
    assert (
        helpers.timestamp_2_gpst_ns(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     6     0     0    1.0000000     GPS"
            )
        )
        == constants.cNanoSecondsPerSecond
    )
    assert (
        helpers.timestamp_2_gpst_ns(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     6     0     0    1.0000001     GPS"
            )
        )
        == constants.cNanoSecondsPerSecond + 100
    )
    assert (
        helpers.timestamp_2_gpst_ns(
            helpers.rinex_header_time_string_2_timestamp_ns(
                "  1980     1     7     0     0    0.0000000     GPS"
            )
        )
        == constants.cSecondsPerDay * constants.cNanoSecondsPerSecond
    )
