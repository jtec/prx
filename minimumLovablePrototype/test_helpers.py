import pytest
import helpers
import constants


def test_rinex_header_time_string_2_gpst():
    assert (
        helpers.rinex_header_time_string_2_gpst_ns(
            "  1980     1     6     0     0    0.0000000     GPS"
        )
        == 0
    )
    assert (
        helpers.rinex_header_time_string_2_gpst_ns(
            "  1980     1     6     0     0    1.0000000     GPS"
        )
        == constants.cNanoSecondsPerSecond
    )
    assert (
        helpers.rinex_header_time_string_2_gpst_ns(
            "  1980     1     6     0     0    1.0000001     GPS"
        )
        == constants.cNanoSecondsPerSecond + 100
    )
    assert (
        helpers.rinex_header_time_string_2_gpst_ns(
            "  1980     1     7     0     0    0.0000000     GPS"
        )
        == constants.cSecondsPerDay * constants.cNanoSecondsPerSecond
    )
