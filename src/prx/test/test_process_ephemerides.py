import math
import pandas as pd
import numpy as np
from pathlib import Path
import process_ephemerides as eph
import helpers
import converters
import constants
import shutil
import pytest
import os


# This function sets up a temporary directory, copies a zipped rinex navigation file into that directory
# and returns its path. The @pytest.fixture annotation allows us to pass the function as an input
# to test functions. When running a test function, pytest will then first run this function, pass
# whatever is passed to `yield` to the test function, and run the code after `yield` after the test,
# even  if the test crashes.
@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    gps_rnx3_nav_test_file = test_directory.joinpath(
        "BRDC00IGS_R_20220010000_01D_GN.zip"
    )
    shutil.copy(
        helpers.prx_root().joinpath(
            f"datasets/TLSE_2022001/{gps_rnx3_nav_test_file.name}"
        ),
        gps_rnx3_nav_test_file,
    )
    assert gps_rnx3_nav_test_file.exists()

    all_constellations_rnx3_nav_test_file = test_directory.joinpath(
        "BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    shutil.copy(
        helpers.prx_root().joinpath(
            f"datasets/TLSE_2022001/{all_constellations_rnx3_nav_test_file.name}"
        ),
        all_constellations_rnx3_nav_test_file,
    )
    assert all_constellations_rnx3_nav_test_file.exists()

    all_constellations_rnx3_obs_test_file = test_directory.joinpath(
        "TLSE00FRA_R_20220010000_01D_30S_MO.zip"
    )
    shutil.copy(
        helpers.prx_root().joinpath(
            f"datasets/TLSE_2022001/{all_constellations_rnx3_obs_test_file.name}"
        ),
        all_constellations_rnx3_obs_test_file,
    )
    assert all_constellations_rnx3_obs_test_file.exists()

    yield {
        "gps_nav_file": gps_rnx3_nav_test_file,
        "all_constellations_nav_file": all_constellations_rnx3_nav_test_file,
        "all_constellations_obs_file": all_constellations_rnx3_obs_test_file,
    }
    shutil.rmtree(test_directory)




def test_galileo_position_and_velocity_sanity_check(input_for_test):
    # Using the following Galileo ephemeris
    """
    E01 2022 01 01 00 30 00-5.500090774149e-04-8.029132914089e-12 0.000000000000e+00
         9.900000000000e+01 1.554687500000e+02 2.642252917583e-09-3.049008177372e+00
         7.327646017075e-06 2.901279367507e-04 1.342035830021e-05 5.440621290207e+03
         5.202000000000e+05 0.000000000000e+00 2.999553080386e+00 1.676380634308e-08
         9.757125000792e-01 6.134375000000e+01-1.229150455736e-01-5.152714631247e-09
        -2.246522148093e-10 2.580000000000e+02 2.190000000000e+03
         3.120000000000e+00 0.000000000000e+00 4.656612873077e-10 0.000000000000e+00
         5.209900000000e+05
    """
    # Copied from the following RINEX navigation file
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        input_for_test["all_constellations_nav_file"]
    )
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # We compute orbital position and velocity of
    sv = "E01"

    # For the following GST time a few minutes after the ephemeris reference time
    gst_week = 2190
    gst_tow = 520200 + 5 * constants.cSecondsPerMinute
    t_orbit_gst_ns = pd.Timedelta(
        gst_week * constants.cSecondsPerWeek + gst_tow, "seconds"
    )

    p_ecef, v_ecef, clock_offset, clock_offset_rate, _ = eph.compute_satellite_state(
        ephemerides, sv, t_orbit_gst_ns
    )

    assert abs(np.linalg.norm(p_ecef) - 29.994 * 1e6) < 1e6
    assert abs(np.linalg.norm(v_ecef) - 3 * 1e3) < 1e2


def test_beidou_position_and_velocity_sanity_check(input_for_test):
    # Using the following Beidou ephemeris
    """
    C01 2022 01 01 00 00 00-2.854013582692e-04 4.026112776501e-11 0.000000000000e+00
         1.000000000000e+00 7.552500000000e+02-4.922705050425e-09 5.928667085353e-01
         2.444302663207e-05 6.108939414844e-04 2.280483022332e-05 6.493410568237e+03
         5.184000000000e+05-2.812594175339e-07-2.969991558287e+00-6.519258022308e-08
         8.077489154703e-02-7.019375000000e+02-1.279165335399e+00 6.122397879589e-09
        -1.074687622196e-09 0.000000000000e+00 8.340000000000e+02 0.000000000000e+00
         2.000000000000e+00 0.000000000000e+00-5.800000000000e-09-1.020000000000e-08
         5.184276000000e+05 0.000000000000e+00
    """
    # Copied from the following RINEX navigation file
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        input_for_test["all_constellations_nav_file"]
    )
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # We compute orbital position and velocity of
    sv = "C01"

    # For the following BDT time a few minutes after the ephemeris reference time
    gst_week = 834
    gst_tow = 518400 + 5 * constants.cSecondsPerMinute
    t_orbit_gst_ns = pd.Timedelta(
        gst_week * constants.cSecondsPerWeek + gst_tow, "seconds"
    )

    p_ecef, v_ecef, clock_offset, clock_offset_rate, _ = eph.compute_satellite_state(
        ephemerides, sv, t_orbit_gst_ns
    )

    # Compare to semi-major-axis:
    assert abs(np.linalg.norm(p_ecef) - 6.493410568237e03**2) < 1e6
    # C01 is on a geosynchronous orbit, so its velocity in ECEF should be smaller than those on MEO orbits
    assert abs(np.linalg.norm(v_ecef) - 2 * 1e2) < 1e1


def test_glonass_position_and_velocity_sanity_check(input_for_test):
    # Using the following GLONASS ephemeris
    """
    R01 2022 01 01 00 15 00 7.305294275284e-06-0.000000000000e+00 5.184000000000e+05
         2.051218896484e+04-9.885606765747e-01 2.793967723846e-09 0.000000000000e+00
         1.273344482422e+04-5.559444427490e-01-3.725290298462e-09 1.000000000000e+00
         8.218795898438e+03 3.327781677246e+00 9.313225746155e-10 0.000000000000e+00
                             .999999999999e+09 1.500000000000e+01
    """
    # Copied from the following RINEX navigation file
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        input_for_test["all_constellations_nav_file"]
    )
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # We compute orbital position and velocity of
    sv = "R01"
    # for the following time
    t_orbit = (
        pd.Timestamp("2022-01-01T00:15:00.000000000")
        - constants.cArbitraryGlonassUtcEpoch
    ) + pd.Timedelta(1, "seconds")

    # there we go:
    p_ecef, v_ecef, _, _, _ = eph.compute_satellite_state(ephemerides, sv, t_orbit)

    assert abs(np.linalg.norm(p_ecef) - 25 * 1e6) < 1e6
    assert abs(np.linalg.norm(v_ecef) - 3.5 * 1e3) < 1e2


def test_compute_gps_satellite_clock_offset(input_for_test):
    # GPS, GAL, QZSS, BDS, IRNSS broadcast satellite clock system time offsets are all given
    # as parameters of a polynomial of order 2, so this test should cover those constellations.
    # When computing the satellite clock offset of GPS-001 for January 1st 2022 at 1am GPST,
    # we expect the clock offset to be computed from the following RINEX 3 ephemeris
    """
    G01 2022 01 01 00 00 00 4.691267386079e-04-1.000444171950e-11 0.000000000000e+00
         3.900000000000e+01-1.411250000000e+02 3.988380417768e-09-6.242942382352e-01
        -7.363036274910e-06 1.121813920327e-02 4.695728421211e-06 5.153674995422e+03
         5.184000000000e+05-3.166496753693e-08-1.036611240093e+00 1.955777406693e-07
         9.864187694897e-01 2.997500000000e+02 8.840876015687e-01-8.133553080847e-09
        -3.778728827795e-10 1.000000000000e+00 2.190000000000e+03 0.000000000000e+00
         2.000000000000e+00 0.000000000000e+00 5.122274160385e-09 3.900000000000e+01
         5.171890000000e+05 4.000000000000e+00 0.000000000000e+00 0.000000000000e+00
    """
    # copied from the following file
    rinex_3_navigation_file = converters.anything_to_rinex_3(
        input_for_test["gps_nav_file"]
    )
    (
        computed_offset_m,
        computed_offset_rate_mps,
    ) = eph.compute_satellite_clock_offset_and_clock_offset_rate(
        eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file),
        "G01",
        helpers.timestamp_2_timedelta(
            pd.Timestamp("2022-01-01T01:00:00.000000000"), "GPST"
        ),
    )
    # We expect the following clock offset and clock offset rate computed by hand from the parameters above.
    delta_t_s = constants.cSecondsPerHour
    expected_offset_m = constants.cGpsSpeedOfLight_mps * (
        4.691267386079e-04
        + (-1.000444171950e-11 * delta_t_s)
        + 0.000000000000e00 * math.pow(delta_t_s, 2)
    )
    expected_offset_rate_mps = constants.cGpsSpeedOfLight_mps * (
        -1.000444171950e-11 + 2 * 0.000000000000e00 * delta_t_s
    )
    # Expect micrometers and micrometers/s accuracy here:
    assert abs(expected_offset_m - computed_offset_m) < 1e-6
    assert abs(expected_offset_rate_mps - computed_offset_rate_mps) < 1e-6


def test_compute_satellite_clock_offset_glonass(input_for_test):
    # Glonass broadcast system time clock offsets are given as a clock offset in seconds
    # plus a relative frequency offset.
    # When computing the satellite clock offset of Glonass-001 for January 1st 2022 at 1am GLONASST
    # We expect the clock offset to be computed from the following RINEX 3 ephemeris
    """
    R01 2022 01 01 00 45 00 7.305294275284E-06-0.000000000000E+00 5.202000000000E+05
         1.799304101562E+04-1.798223495483E+00 1.862645149231E-09 0.000000000000E+00
         1.165609716797E+04-5.995044708252E-01-3.725290298462E-09 1.000000000000E+00
         1.381343408203E+04 2.848098754883E+00 0.000000000000E+00 0.000000000000E+00
    """
    # copied from the following file
    rinex_3_navigation_file = converters.anything_to_rinex_3(
        input_for_test["all_constellations_nav_file"]
    )
    (
        computed_offset_m,
        computed_offset_rate_mps,
    ) = eph.compute_satellite_clock_offset_and_clock_offset_rate(
        eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file),
        "R01",
        helpers.timestamp_2_timedelta(
            pd.Timestamp("2022-01-01T01:00:00.000000000"), "GLONASST"
        ),
    )
    # We expect the following clock offset and clock offset rate computed by hand from the parameters above.
    delta_t_s = constants.cSecondsPerHour
    expected_offset_m = constants.cGpsSpeedOfLight_mps * (
        7.305294275284e-06 + (0.0 * delta_t_s) + math.pow(0.000000000000e00, 2)
    )
    expected_offset_rate_mps = 0
    # Expect micrometers and micrometers/s accuracy here:
    assert abs(expected_offset_m - computed_offset_m) < 1e-6
    assert abs(expected_offset_rate_mps - computed_offset_rate_mps) < 1e-6


def test_sagnac_effect():
    # load validation data
    path_to_validation_file = helpers.prx_root().joinpath(
        f"tools/validation_data/sagnac_effect.csv"
    )
    # read satellite position in header line 2
    sat_pos = np.array((28400000, 0, 0))

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
        sagnac_effect_computed[ind] = eph.compute_sagnac_effect(sat_pos, rx_pos)

    # errors come from the approximation of cos and sin for small angles
    # millimeter accuracy should be sufficient
    tolerance = 1e-3
    assert np.max(np.abs(sagnac_effect_computed - sagnac_effect_reference)) < tolerance


def test_ecef_to_geodetic():
    tolerance_rad = 1e-3 / 6400e3  # equivalent to one mm at Earth surface
    tolerance_alt = 1e-3

    ecef_coords = [6378137.0, 0.0, 0.0]
    expected_geodetic = [0.0, 0.0, 0.0]
    computed_geodetic = eph.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [0.0, 6378137.0, 0.0]
    expected_geodetic = [np.deg2rad(0.0), np.deg2rad(90), 0.0]
    computed_geodetic = eph.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [4624518, 116590, 4376497]  # Toulouse, France
    expected_geodetic = [
        np.deg2rad(43.604698100243851),
        np.deg2rad(1.444193786348353),
        151.9032,
    ]
    computed_geodetic = eph.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [
        -4.646050004314417e06,
        2.553206120634516e06,
        -3.534374202256767e06,
    ]  # Sidney
    expected_geodetic = [np.deg2rad(-33.8688197), np.deg2rad(151.2092955), 0]
    computed_geodetic = eph.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt

    ecef_coords = [
        1.362205559782862e06,
        -3.423584689115747e06,
        -5.188704112366104e06,
    ]  # Ushuaia, Argentina
    expected_geodetic = [np.deg2rad(-54.8019121), np.deg2rad(-68.3029511), 0]
    computed_geodetic = eph.ecef_2_geodetic(ecef_coords)
    assert (
        np.abs(np.array(expected_geodetic[:2]) - np.array(computed_geodetic[:2]))
        < tolerance_rad
    ).all()
    assert np.abs(expected_geodetic[2] - computed_geodetic[2]) < tolerance_alt


def test_satellite_elevation_and_azimuth():
    tolerance = np.deg2rad(1e-3)

    sat_pos_ecef = np.array([26600e3, 0.0, 0.0])
    rx_pos_ecef = np.array([6400e3, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(90), np.deg2rad(0)
    computed_el, computed_az = eph.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array([2.066169397996826e07, 0.0, 1.428355697996826e07])
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(0)
    computed_el, computed_az = eph.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance

    sat_pos_ecef = np.array(
        [2.066169397996826e07, 7.141778489984130e06, 1.236992320105505e07]
    )
    rx_pos_ecef = np.array([6378137.0, 0.0, 0.0])
    expected_el, expected_az = np.deg2rad(45), np.deg2rad(30)
    computed_el, computed_az = eph.compute_satellite_elevation_and_azimuth(
        sat_pos_ecef, rx_pos_ecef
    )
    assert np.abs(expected_el - computed_el) < tolerance
    assert np.abs(expected_az - computed_az) < tolerance
