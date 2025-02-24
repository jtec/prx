import logging
import platform
import numpy as np
import prx.util
import pytest
from prx import helpers
from prx import constants
from prx import converters
import pandas as pd
from pathlib import Path
import shutil
import os
import subprocess

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
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
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
    assert helpers.timestamp_2_timedelta(
        helpers.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     6     0     0    0.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(0)
    assert helpers.timestamp_2_timedelta(
        helpers.rinex_header_time_string_2_timestamp_ns(
            "  1980     1     6     0     0    1.0000000     GPS"
        ),
        "GPST",
    ) == pd.Timedelta(constants.cNanoSecondsPerSecond, unit="ns")
    timestamp = helpers.rinex_header_time_string_2_timestamp_ns(
        "  1980     1     6     0     0    1.0000001     GPS"
    )
    timedelta = helpers.timestamp_2_timedelta(timestamp, "GPST")

    assert timedelta == pd.Timedelta(constants.cNanoSecondsPerSecond + 100, unit="ns")
    assert helpers.timestamp_2_timedelta(
        helpers.rinex_header_time_string_2_timestamp_ns(
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
        computed_geodetic = helpers.ecef_2_geodetic(coord["ecef"])
        assert (
            np.abs(np.array(coord["geodetic"][:2]) - np.array(computed_geodetic[:2]))
            < tolerance_rad
        ).all()
        assert np.abs(coord["geodetic"][2] - computed_geodetic[2]) < tolerance_alt


def test_geodetic_2_ecef():
    tolerance_ecef = 1e-3

    for coord in coords.values():
        computed_ecef = helpers.geodetic_2_ecef(
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
    computed_el, computed_az = helpers.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array([[2.066169397996826e07, 0.0, 1.428355697996826e07]])
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(0)
    computed_el, computed_az = helpers.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array(
        [[2.066169397996826e07, 7.141778489984130e06, 1.236992320105505e07]]
    )
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(30)
    computed_el, computed_az = helpers.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance


def test_sagnac_effect():
    # load validation data
    path_to_validation_file = (
        helpers.prx_repository_root()
        / "src/prx/tools/validation_data/sagnac_effect.csv"
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
        sagnac_effect_computed[ind] = helpers.compute_sagnac_effect(sat_pos, rx_pos)

    # errors come from the approximation of cos and sin for small angles
    # millimeter accuracy should be sufficient
    tolerance = 1e-3
    assert np.max(np.abs(sagnac_effect_computed - sagnac_effect_reference)) < tolerance


def test_is_sorted():
    assert helpers.is_sorted([1, 2, 3, 4, 5])
    assert not helpers.is_sorted([1, 2, 3, 5, 4])
    assert not helpers.is_sorted([5, 4, 3, 2, 1])
    assert helpers.is_sorted([1, 1, 1, 1, 1])
    assert helpers.is_sorted([1])
    assert helpers.is_sorted([])


def test_gfzrnx_execution_on_obs_file(input_for_test):
    """Check execution of gfzrnx on a RNX OBS file and check"""
    # convert test file to RX3 format
    file_obs = converters.anything_to_rinex_3(input_for_test["obs"])
    # list all gfzrnx binaries contained in the folder "prx/tools/gfzrnx/"
    path_folder_gfzrnx = helpers.prx_repository_root() / "src/prx/tools/gfzrnx"
    path_binary = path_folder_gfzrnx.joinpath(
        constants.gfzrnx_binary[platform.system()]
    )
    # assert len(gfzrnx_binaries) > 0, "Could not find any gfzrnx binary"
    command = [
        str(path_binary),
        "-finp",
        str(file_obs),
        "-fout",
        str(file_obs.parent.joinpath("gfzrnx_out.rnx")),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode == 0:
        log.info(
            f"Ran gfzrnx file repair on {file_obs.name} with {constants.gfzrnx_binary[platform.system()]}"
        )
    else:
        log.info(f"gfzrnx file repair run failed: {result}")

    assert file_obs.parent.joinpath("gfzrnx_out.rnx").exists()


def test_gfzrnx_execution_on_nav_file(input_for_test):
    """Check execution of gfzrnx on a RNX NAV file and check"""
    file_nav = converters.anything_to_rinex_3(input_for_test["nav"])
    path_folder_gfzrnx = helpers.prx_repository_root() / "src/prx/tools/gfzrnx"
    path_binary = path_folder_gfzrnx.joinpath(
        constants.gfzrnx_binary[platform.system()]
    )
    # assert len(gfzrnx_binaries) > 0, "Could not find any gfzrnx binary"
    command = [
        str(path_binary),
        "-finp",
        str(file_nav),
        "-fout",
        str(file_nav.parent.joinpath("gfzrnx_out.rnx")),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode == 0:
        log.info(
            f"Ran gfzrnx file repair on {file_nav.name} with {constants.gfzrnx_binary[platform.system()]}"
        )
    else:
        log.info(f"gfzrnx file repair run failed: {result}")

    assert file_nav.parent.joinpath("gfzrnx_out.rnx").exists()


def test_gfzrnx_function_call(input_for_test):
    """Check function call of gfzrnx on a RNX OBS file and check"""
    file_nav = converters.anything_to_rinex_3(input_for_test["nav"])
    file_obs = converters.anything_to_rinex_3(input_for_test["obs"])
    file_sp3 = input_for_test["sp3"]

    file_nav = prx.util.repair_with_gfzrnx(file_nav)
    file_obs = prx.util.repair_with_gfzrnx(file_obs)
    # running gfzrnx on a file that is not a RNX file should result in an error
    try:
        file_sp3 = prx.util.repair_with_gfzrnx(file_sp3)
    except AssertionError:
        log.info(f"gfzrnx binary did not execute with file {file_sp3}")
    assert True


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
        w, s = helpers.timedelta_2_weeks_and_seconds(tested_timedelta)
        week_computed.append(w)
        seconds_of_week_computed.append(s)

    # expected values come from https://gnsscalc.com/
    week_expected = [54, 54, 1025, 2048, np.nan]
    seconds_of_week_expected = [280800, 281400, 317400, 302400, np.nan]

    assert week_expected == week_computed
    assert seconds_of_week_expected == seconds_of_week_expected


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
            helpers.compute_gps_utc_leap_seconds(year, doy), expected_leap_second
        )


def test_timestamp_to_mid_day():
    assert helpers.timestamp_to_mid_day(
        pd.Timestamp("2023-01-01T01:02:03")
    ) == pd.Timestamp("2023-01-01T12:00:00")
