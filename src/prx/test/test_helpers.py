import logging
import numpy as np
import pytest
from prx import constants, util
import pandas as pd
from pathlib import Path
import shutil
import os

log = logging.getLogger(__name__)


@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    test_file = {"obs": test_directory.joinpath(compressed_compact_rinex_file)}
    shutil.copy(
        Path(__file__).parent
        / f"datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_file["obs"],
    )
    assert test_file["obs"].exists()

    # Also provide ephemerides so the test does not have to download them:
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.gz"
    test_file["nav"] = test_directory.joinpath(ephemerides_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{ephemerides_file}",
        test_file["nav"].parent.joinpath(ephemerides_file),
    )
    assert test_file["nav"].parent.joinpath(ephemerides_file).exists()

    # sp3 file
    sp3_file = "GFZ0MGXRAP_20230010000_01D_05M_ORB.SP3"
    test_file["sp3"] = test_directory.joinpath(sp3_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{sp3_file}",
        test_file["sp3"].parent.joinpath(sp3_file),
    )
    assert test_file["sp3"].parent.joinpath(sp3_file).exists()

    yield test_file
    shutil.rmtree(test_directory)


coords = {
    "equator_0": {"ecef": [6378137.0, 0.0, 0.0], "geodetic": [0.0, 0.0, 0.0]},
    "equator_90": {
        "ecef": [0.0, 6378137.0, 0.0],
        "geodetic": [np.deg2rad(0.0), np.deg2rad(90), 0.0],
    },
    "toulouse": {
        "ecef": [4624518, 116590, 4376497],
        "geodetic": [np.deg2rad(43.604698100243), np.deg2rad(1.444193786348), 151.9032],
    },
    "Sidney": {
        "ecef": [-4.646050004314417e06, 2.553206120634516e06, -3.534374202256767e06],
        "geodetic": [np.deg2rad(-33.8688197), np.deg2rad(151.2092955), 0],
    },
    "Ushuaia": {
        "ecef": [1.362205559782862e06, -3.423584689115747e06, -5.188704112366104e06],
        "geodetic": [np.deg2rad(-54.8019121), np.deg2rad(-68.3029511), 0],
    },
}


def test_rinex_header_time_string_2_timestamp_ns():
    assert util.timestamp_2_timedelta(
        util.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     6     0     0    0.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(0)
    assert util.timestamp_2_timedelta(
        util.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     6     0     0    1.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(constants.cNanoSecondsPerSecond, unit="ns")
    timestamp = util.rinex_header_time_string_2_timestamp_ns(
        "  1980     1     6     0     0    1.0000001     GPS"
    )
    timedelta = util.timestamp_2_timedelta(timestamp, "GPST")

    assert timedelta == pd.Timedelta(constants.cNanoSecondsPerSecond + 100, unit="ns")
    assert util.timestamp_2_timedelta(
        util.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     7     0     0    0.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(
        constants.cSecondsPerDay * constants.cNanoSecondsPerSecond, unit="ns"
    )


def test_ecef_to_geodetic():
    tolerance_rad = 1e-3 / 6400e3  # equivalent to one mm at Earth surface
    tolerance_alt = 1e-3

    for coord in coords.values():
        computed_geodetic = util.ecef_2_geodetic(coord["ecef"])
        assert (
            np.abs(np.array(coord["geodetic"][:2]) - np.array(computed_geodetic[:2]))
            < tolerance_rad
        ).all()
        assert np.abs(coord["geodetic"][2] - computed_geodetic[2]) < tolerance_alt


def test_geodetic_2_ecef():
    tolerance_ecef = 1e-3

    for coord in coords.values():
        computed_ecef = util.geodetic_2_ecef(
            coord["geodetic"][0], coord["geodetic"][1], coord["geodetic"][2]
        )
        assert (
            np.linalg.norm(np.array(coord["ecef"]) - np.array(computed_ecef))
            < tolerance_ecef
        )


def test_satellite_elevation_and_azimuth():
    tolerance = np.deg2rad(1e-3)

    sat_pos_ecef = np.array([[26600e3, 0.0, 0.0]])
    rx_pos_ecef = np.array([6400e3, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(90), np.deg2rad(0)
    computed_el, computed_az = util.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array([[2.066169397996826e07, 0.0, 1.428355697996826e07]])
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(0)
    computed_el, computed_az = util.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array(
        [[2.066169397996826e07, 7.141778489984130e06, 1.236992320105505e07]]
    )
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(30)
    computed_el, computed_az = util.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance


def test_sagnac_effect():
    # load validation data
    path_to_validation_file = (
        util.prx_repository_root() / "src/prx/tools/validation_data/sagnac_effect.csv"
    )

    # satellite position (from reference CSV header)
    sat_pos = np.array([[28400000, 0, 0]])

    # read data
    data = np.loadtxt(
        path_to_validation_file,
        delimiter=",",
        skiprows=3,
    )

    sagnac_effect_reference = np.zeros((data.shape[0],))
    sagnac_effect_computed = np.zeros((data.shape[0],))
    for ind in range(data.shape[0]):
        rx_pos = data[ind, 0:3]
        sagnac_effect_reference[ind] = data[ind, 3]
        sagnac_effect_computed[ind] = util.compute_sagnac_effect(sat_pos, rx_pos)

    # errors come from the approximation of cos and sin for small angles
    # millimeter accuracy should be sufficient
    tolerance = 1e-3
    assert np.max(np.abs(sagnac_effect_computed - sagnac_effect_reference)) < tolerance


def test_is_sorted():
    assert util.is_sorted([1, 2, 3, 4, 5])
    assert not util.is_sorted([1, 2, 3, 5, 4])
    assert not util.is_sorted([5, 4, 3, 2, 1])
    assert util.is_sorted([1, 1, 1, 1, 1])
    assert util.is_sorted([1])
    assert util.is_sorted([])


def test_row_wise_dot_product():
    # Check whether the way we compute the row-wise dot product with numpy yields the expected result
    A = np.array([[1, 2], [4, 5], [7, 8]])
    B = np.array([[10, 20], [30, 40], [50, 60]])
    row_wise_dot = np.sum(A * B, axis=1).reshape(-1, 1)
    assert (row_wise_dot == np.array([[10 + 40], [120 + 200], [350 + 480]])).all()


def test_timedelta_2_weeks_and_seconds():
    tested_timestamps = [
        pd.Timestamp("1981-01-21T06:00:00"),
        pd.Timestamp("1981-01-21T06:10:00"),
        pd.Timestamp("1999-09-01T16:10:00"),  # after first week number rollover
        pd.Timestamp("2019-04-10T12:00:00"),  # after 2nd week number rollover
        pd.Timestamp(""),  # yields a NaT
    ]

    week_computed = []
    seconds_of_week_computed = []
    for timestamp in tested_timestamps:
        tested_timedelta = (
            timestamp - constants.system_time_scale_rinex_utc_epoch["GPST"]
        )
        w, s = util.timedelta_2_weeks_and_seconds(tested_timedelta)
        week_computed.append(w)
        seconds_of_week_computed.append(s)

    # expected values come from https://gnsscalc.com/
    week_expected = [54, 54, 1025, 2048, np.nan]
    seconds_of_week_expected = [280800, 281400, 317400, 302400, np.nan]

    np.testing.assert_array_equal(week_computed, week_expected)
    np.testing.assert_array_equal(seconds_of_week_computed, seconds_of_week_expected)

    # We also expect the function to work for Series of timestamps
    week_series_computed, seconds_of_week_series_computed = (
        util.timedelta_2_weeks_and_seconds(
            pd.Series(tested_timestamps)
            - constants.system_time_scale_rinex_utc_epoch["GPST"]
        )
    )
    np.testing.assert_array_equal(week_series_computed, week_expected)
    np.testing.assert_array_equal(
        seconds_of_week_series_computed, seconds_of_week_series_computed
    )


def test_compute_gps_leap_seconds():
    # expected GPS leap second come from https://gnsscalc.com/
    test_cases = [
        (1982, 181, 2),  # year, doy (corresponds to 01-Jul), expected leap seconds
        (1989, 181, 5),
        (1992, 182, 8),  # leap year, still corresponds to 01-Jul
        (2008, 182, 14),
        (2016, 182, 17),
        (2022, 181, 18),
        (1979, 1, np.nan),
    ]
    for year, doy, expected_leap_second in test_cases:
        np.testing.assert_equal(
            util.compute_gps_utc_leap_seconds(year, doy), expected_leap_second
        )


def test_timestamp_to_mid_day():
    assert util.timestamp_to_mid_day(
        pd.Timestamp("2023-01-01T01:02:03")
    ) == pd.Timestamp("2023-01-01T12:00:00")


def test_sun_pos():
    pos_sun = (
        util.compute_sun_ecef_position(
            np.array(
                [
                    pd.Timestamp(
                        year=2020, month=6, day=20, hour=6
                    ),  # approx time when the sun is "close" to [0,1,xx] AU (Equinox)pd.Timestamp(
                    pd.Timestamp(
                        year=2020, month=6, day=20, hour=12
                    ),  # approx time when the sun is "close" to [1,0,xx] AU (Equinox)
                    pd.Timestamp(
                        year=2020, month=6, day=20, hour=18
                    ),  # approx time when the sun is "close" to [0,-1,xx] AU
                ]
            )
        )
        / np.array([149_597_870_700] * 3)
    )
    assert pos_sun[0:2, 0] == pytest.approx(np.array([0, 1]), abs=0.1)
    assert pos_sun[0:2, 1] == pytest.approx(np.array([1, 0]), abs=0.1)
    assert pos_sun[0:2, 2] == pytest.approx(np.array([0, -1]), abs=0.1)


def test_sat_frame():
    """
    Some tests to validate the conversion between ECEF and satellite-fixed frame.

    In order to control the sun position, the function util.ecef_2_satellite is not used, but its content is
    replicated here.
    """
    pos_sat_ecef = np.array([26_600_000, 0, 0]).reshape(1, -1)
    pos_ecef = np.array([6_400_000, 0, 0]).reshape(1, -1)

    k = -pos_sat_ecef / np.linalg.norm(pos_sat_ecef, axis=1).reshape(1, -1)
    pos_sun_ecef = np.array([0, 149_597_870_700, 0]).reshape(1, -1)  # 1 AU on y
    unit_vector_sun_ecef = (pos_sun_ecef - pos_sat_ecef) / np.linalg.norm(
        pos_sun_ecef - pos_sat_ecef, axis=1
    ).reshape(-1, 1)
    j = np.cross(k, unit_vector_sun_ecef)
    i = np.cross(j, k)

    rot_mat_ecef2sat = np.stack([i, j, k], axis=1)
    pos_sat_frame = np.stack(
        [
            rot_mat_ecef2sat[i, :, :] @ (pos_ecef[i, :] - pos_sat_ecef[i, :])
            for i in range(pos_sat_ecef.shape[0])
        ]
    )
    assert pos_sat_frame[0, :] == pytest.approx(np.array([0, 0, 20_200_000]))

    # other simple examples
    pos_sun = np.array([0, 149_597_870_700, 0])  # 1 AU on y
    pos_rx = np.array([6_400_000, 0, 0])  # on +x
    pos_sat = np.array([26_600_000, 0, 0])  # on +x

    k = -pos_sat / np.linalg.norm(pos_sat)
    unit_vector_sun_ecef = (pos_sun - pos_sat) / np.linalg.norm(pos_sun - pos_sat)
    j = np.cross(k, unit_vector_sun_ecef)
    i = np.cross(j, k)

    pos_rx_sat = np.stack([i, j, k], axis=0) @ (pos_rx - pos_sat)
    assert (pos_rx_sat == np.array([0, 0, 20_200_000])).all()

    pos_rx = np.array([0, 6_400_000, 0])  # on +y
    pos_rx_sat = np.stack([i, j, k], axis=0) @ (pos_rx - pos_sat)
    assert pos_rx_sat == pytest.approx(np.array([6_400_000, 0, 20_200_000 + 6_400_000]))
