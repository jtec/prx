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
    gps_rnx3_nav_test_file = test_directory.joinpath("BRDC00IGS_R_20220010000_01D_GN.zip")
    shutil.copy(
        helpers.prx_root().joinpath(
            f"datasets/TLSE_2022001/{gps_rnx3_nav_test_file.name}"
        ),
        gps_rnx3_nav_test_file,
    )
    assert gps_rnx3_nav_test_file.exists()

    all_constellations_rnx3_nav_test_file = test_directory.joinpath("BRDC00IGS_R_20220010000_01D_MN.zip")
    shutil.copy(
        helpers.prx_root().joinpath(
            f"datasets/TLSE_2022001/{all_constellations_rnx3_nav_test_file.name}"
        ),
        all_constellations_rnx3_nav_test_file,
    )
    assert all_constellations_rnx3_nav_test_file.exists()

    yield {"gps_nav_file": gps_rnx3_nav_test_file, "all_constellations_nav_file": all_constellations_rnx3_nav_test_file}
    shutil.rmtree(test_directory)


def test_compare_rnx3_gps_sat_pos_with_magnitude(input_for_test):
    """Loads a RNX3 file, compute a position for different satellites and time, and compare to MAGNITUDE results
    Test will be a success if the difference in position is lower than threshold_pos_error_m = 0.01
    """
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # select sv and time
    sv = "G01"
    gpst_week = 2190
    gpst_tow = 523800


    # MAGNITUDE position
    sv_pos_magnitude = np.array([13053451.235, -12567273.060, 19015357.126])

    # Compute broadcast satellite position
    # Select right ephemeris
    t_gpst_for_which_to_compute_position = pd.Timedelta(gpst_week*constants.cSecondsPerWeek + gpst_tow, "seconds")
    p_ecef, v_ecef, _, _  = eph.compute_satellite_state(ephemerides, sv, t_gpst_for_which_to_compute_position)

    threshold_pos_error_m = 0.01
    assert np.linalg.norm(p_ecef - sv_pos_magnitude) < threshold_pos_error_m


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
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # We compute orbital position and velocity of
    sv = "E01"

    # For the following GST time a few minutes after the ephemeris reference time
    gst_week = 2190
    gst_tow = 520200 + 5 * constants.cSecondsPerMinute
    t_orbit_gst_ns = pd.Timedelta(gst_week*constants.cSecondsPerWeek + gst_tow, "seconds")

    p_ecef, v_ecef, clock_offset, clock_offset_rate = eph.compute_satellite_state(ephemerides, sv, t_orbit_gst_ns)

    assert abs(np.linalg.norm(p_ecef) - 29.994 * 1e6) < 1e6
    assert abs(np.linalg.norm(v_ecef) - 3*1e3) < 1e2


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
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # We compute orbital position and velocity of
    sv = "C01"

    # For the following BDT time a few minutes after the ephemeris reference time
    gst_week = 834
    gst_tow = 518400 + 5 * constants.cSecondsPerMinute
    t_orbit_gst_ns = pd.Timedelta(gst_week * constants.cSecondsPerWeek + gst_tow, "seconds")

    p_ecef, v_ecef, clock_offset, clock_offset_rate = eph.compute_satellite_state(ephemerides, sv, t_orbit_gst_ns)

    # Compare to semi-major-axis:
    assert abs(np.linalg.norm(p_ecef) - (6.493410568237e+03)**2) < 1e6
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
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    ephemerides = eph.convert_rnx3_nav_file_to_dataframe(path_to_rnx3_nav_file)

    # We compute orbital position and velocity of
    sv = "R01"
    # for the following time
    t_orbit = (pd.Timestamp("2022-01-01T00:15:00.000000000") - constants.cArbitraryGlonassUtcEpoch) + pd.Timedelta(1, "seconds")

    # there we go:
    p_ecef, v_ecef, _, _ = eph.compute_satellite_state(ephemerides, sv, t_orbit)

    assert abs(np.linalg.norm(p_ecef) - 25 * 1e6) < 1e6
    assert abs(np.linalg.norm(v_ecef) - 3.5*1e3) < 1e2


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
    rinex_3_navigation_file = converters.anything_to_rinex_3(input_for_test["gps_nav_file"])
    (
        computed_offset_m,
        computed_offset_rate_mps,
    ) = eph.compute_satellite_clock_offset_and_clock_offset_rate(
        eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file),
        "G01",
        helpers.timestamp_2_timedelta(
        pd.Timestamp("2022-01-01T01:00:00.000000000"), 'GPST'),
    )
    # We expect the following clock offset and clock offset rate computed by hand from the parameters above.
    delta_t_s = constants.cSecondsPerHour
    expected_offset_m = constants.cGpsIcdSpeedOfLight_mps * (
        4.691267386079e-04
        + (-1.000444171950e-11 * delta_t_s)
        + 0.000000000000e00 * math.pow(delta_t_s, 2)
    )
    expected_offset_rate_mps = constants.cGpsIcdSpeedOfLight_mps * (-1.000444171950e-11 + 2 * 0.000000000000e00 * delta_t_s)
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
    rinex_3_navigation_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    (
        computed_offset_m,
        computed_offset_rate_mps,
    ) = eph.compute_satellite_clock_offset_and_clock_offset_rate(
        eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file),
        "R01",
        helpers.timestamp_2_timedelta(
        pd.Timestamp("2022-01-01T01:00:00.000000000"), "GLONASST")
    )
    # We expect the following clock offset and clock offset rate computed by hand from the parameters above.
    delta_t_s = constants.cSecondsPerHour
    expected_offset_m = constants.cGpsIcdSpeedOfLight_mps * (
        7.305294275284e-06 + (0.0 * delta_t_s) + math.pow(0.000000000000e00, 2)
    )
    expected_offset_rate_mps = 0
    # Expect micrometers and micrometers/s accuracy here:
    assert (
        abs(expected_offset_m - computed_offset_m) < 1e-6
    )
    assert (
        abs(expected_offset_rate_mps - computed_offset_rate_mps) < 1e-6
    )


def test_compute_gps_group_delay_rnx3(input_for_test):
    """
    Computes the total group delay (TGD) from a RNX3 NAV file containing the following ephemerides. The TGD is
    highlighted between **

    This tests also validates
    - the choice of the right ephemeris for the correct time: 3 epochs are used
    - the scaling of the tgd with the carrier frequency: the 3 observations types considered in IS-GPS-200N are tested (
    C1C, C1P, C2P) and 1 not considered shall return NaN (C1Y)

    G02 2022 01 01 00 00 00-6.473939865830e-04-1.136868377220e-12 0.000000000000e+00
         4.100000000000e+01-1.427187500000e+02 4.556261215140e-09-4.532451297190e-01
        -7.487833499910e-06 2.063889056440e-02 4.231929779050e-06 5.153668174740e+03
         5.184000000000e+05-2.589076757430e-07-1.124962932550e+00 1.229345798490e-07
         9.647440889150e-01 3.036250000000e+02-1.464769118290e+00-8.689647672990e-09
        -2.621537769000e-10 1.000000000000e+00 2.190000000000e+03 0.000000000000e+00
         2.000000000000e+00 0.000000000000e+00**-1.769512891770e-08** 4.100000000000e+01
         5.112180000000e+05 4.000000000000e+00 0.000000000000e+00 0.000000000000e+00
    G02 2022 01 01 02 00 00-6.474019028246e-04-1.136868377216e-12 0.000000000000e+00
         4.200000000000e+01-1.350312500000e+02 4.654122434309e-09 5.969613050574e-01
        -7.288530468941e-06 2.063782420009e-02 4.164874553680e-06 5.153666042328e+03
         5.256000000000e+05-2.495944499969e-07-1.125024713041e+00-1.173466444016e-07
         9.647413912942e-01 3.089375000000e+02-1.464791005008e+00-8.692504934864e-09
        -4.328751738457e-10 1.000000000000e+00 2.190000000000e+03 0.000000000000e+00
         2.000000000000e+00 0.000000000000e+00**-1.769512891769e-08** 4.200000000000e+01
         5.184180000000e+05 4.000000000000e+00 0.000000000000e+00 0.000000000000e+00
    """
    # parse rinex3 nav file
    rinex_3_navigation_file = converters.anything_to_rinex_3(input_for_test["gps_nav_file"])
    eph_rnx3_df = eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file)

    # retrieve total group delays for 4 different observation codes, at 3 different times
    tgd_c1c_s = pd.Series(data=[
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:00:00.000000000"), "GPST"),
                                           "G02", "C1C"),
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GPST"),
                                           "G02", "C1C"),
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T02:15:00.000000000"), "GPST"),
                                           "G02", "C1C"),
    ])
    tgd_c1p_s = pd.Series(data=[
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:00:00.000000000"), "GPST"),
                                           "G02", "C1P"),
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GPST"),
                                           "G02", "C1P"),
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T02:15:00.000000000"), "GPST"),
                                           "G02", "C1P"),
    ])
    tgd_c2p_s = pd.Series(data=[
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:00:00.000000000"), "GPST"),
                                           "G02", "C2P"),
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GPST"),
                                           "G02", "C2P"),
        eph.compute_total_group_delay_rnx3(eph_rnx3_df, helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T02:15:00.000000000"), "GPST"),
                                           "G02", "C2P"),
    ])
    tgd_c5x_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:00:00.000000000"), "GPST"), "G02",
                                                   "C5X")

    # total group delay is on the 7th line, 3rd position
    tgd_c1c_s_expected = pd.Series(data=[-1.769512891770e-08, -1.769512891770e-08, -1.769512891769e-08])
    tgd_c1p_s_expected = pd.Series(data=[-1.769512891770e-08, -1.769512891770e-08, -1.769512891769e-08])
    tgd_c2p_s_expected = pd.Series(data=[-1.769512891770e-08, -1.769512891770e-08, -1.769512891769e-08]) * \
                         (constants.carrier_frequencies_hz()["G"]["L1"] / constants.carrier_frequencies_hz()["G"]["L2"]) ** 2

    assert (tgd_c1c_s == tgd_c1c_s_expected).all()
    assert (tgd_c1p_s == tgd_c1p_s_expected).all()
    assert (tgd_c2p_s == tgd_c2p_s_expected).all()
    assert (np.isnan(tgd_c5x_s))


def test_compute_gal_group_delay_rnx3(input_for_test):
    """
    E25 2022 01 01 00 00 00-5.587508785538e-04-1.278976924368e-12 0.000000000000e+00
         9.600000000000e+01 1.960625000000e+02 2.576178736756e-09 1.034878564309e+00
         9.007751941681e-06 3.147422103211e-04 1.221708953381e-05 5.440630062103e+03
         5.184000000000e+05 1.303851604462e-08 2.996653673305e+00 9.313225746155e-09
         9.752953840989e-01 8.875000000000e+01-5.349579038983e-01-5.132356640398e-09
        -2.260808457461e-10 2.580000000000e+02 2.190000000000e+03 0.000000000000e+00
         3.120000000000e+00 0.000000000000e+00 **4.423782229424e-09** 0.000000000000e+00
         5.191430000000e+05 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
    E25 2022 01 01 00 00 00-5.587504128930e-04-1.278976924370e-12 0.000000000000e+00
         9.600000000000e+01 1.960625000000e+02 2.576178736760e-09 1.034878564310e+00
         9.007751941680e-06 3.147422103210e-04 1.221708953380e-05 5.440630062100e+03
         5.184000000000e+05 1.303851604460e-08 2.996653673310e+00 9.313225746150e-09
         9.752953840990e-01 8.875000000000e+01-5.349579038980e-01-5.132356640400e-09
        -2.260808457460e-10 5.160000000000e+02 2.190000000000e+03
         3.120000000000e+00 0.000000000000e+00 4.423782229420e-09 **4.889443516730e-09**
         5.190640000000e+05
     """
    # parse rinex3 nav file
    rinex_3_navigation_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    eph_rnx3_df = eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file)

    tgd_e1_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                  helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GST"), "E25",
                                                  "C1C")
    tgd_e5a_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GST"), "E25",
                                                   "C5X")
    tgd_e5b_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GST"), "E25",
                                                   "C7X")
    tgd_e6b_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GST"), "E25",
                                                   "C6B")

    tgd_e1_s_expected = 4.889443516730e-09
    tgd_e5a_s_expected = 4.423782229424e-09 * \
                         (constants.carrier_frequencies_hz()["E"]["L1"] / constants.carrier_frequencies_hz()["E"]["L5"]) ** 2
    tgd_e5b_s_expected = 4.889443516730e-09 * \
                         (constants.carrier_frequencies_hz()["E"]["L1"] / constants.carrier_frequencies_hz()["E"]["L7"]) ** 2

    assert (abs(tgd_e1_s - tgd_e1_s_expected) * constants.cGpsIcdSpeedOfLight_mps < 1e-3)
    assert (abs(tgd_e5a_s - tgd_e5a_s_expected) * constants.cGpsIcdSpeedOfLight_mps < 1e-3)
    assert (abs(tgd_e5b_s - tgd_e5b_s_expected) * constants.cGpsIcdSpeedOfLight_mps < 1e-3)
    assert np.isnan(tgd_e6b_s)


def test_compute_bds_group_delay_rnx3(input_for_test):
    """
    C01 2022 01 01 00 00 00-2.854013582692E-04 4.026112776501E-11 0.000000000000E+00
         1.000000000000E+00 7.552500000000E+02-4.922705050425E-09 5.928667085353E-01
         2.444302663207E-05 6.108939414844E-04 2.280483022332E-05 6.493410568237E+03
         5.184000000000E+05-2.812594175339E-07-2.969991558287E+00-6.519258022308E-08
         8.077489154703E-02-7.019375000000E+02-1.279165335399E+00 6.122397879589E-09
        -1.074687622196E-09 0.000000000000E+00 8.340000000000E+02 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00**-5.800000000000E-09-1.020000000000E-08**
         5.184276000000E+05 0.000000000000E+00
    C01 2022 01 01 01 00 00-2.852565376088E-04 4.024691691029E-11 0.000000000000E+00
         1.000000000000E+00 9.752656250000E+02-8.727863550550E-09 8.501598658737E-01
         3.171199932694E-05 6.130223628134E-04 5.824025720358E-06 6.493419839859E+03
         5.220000000000E+05-2.081505954266E-07-2.690127624533E+00 4.563480615616E-08
         8.254541302207E-02-1.764687500000E+02-1.553735793709E+00 9.873625561851E-09
        -8.761079219830E-10 0.000000000000E+00 8.340000000000E+02 0.000000000000E+00
         2.000000000000E+00 0.000000000000E+00**-5.800000000000E-09-1.020000000000E-08**
         5.220276000000E+05 0.000000000000E+00
    """
    # parse rinex3 nav file
    rinex_3_navigation_file = converters.anything_to_rinex_3(input_for_test["all_constellations_nav_file"])
    eph_rnx3_df = eph.convert_rnx3_nav_file_to_dataframe(rinex_3_navigation_file)

    # B1I -> C2I
    tgd_c2i_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"), "C01",
                                                   "C2I")
    # B2I -> C7I
    tgd_c7i_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"), "C01",
                                                   "C7I")
    # B3I -> C6I
    tgd_c6i_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"), "C01",
                                                   "C6I")

    # B1Cd -> C1D
    tgd_c1d_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"), "C01",
                                                   "C1D")
    # B1Cp -> C1P
    tgd_c1p_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"), "C01",
                                                   "C1P")
    # B2bi -> C7D
    tgd_c5x_s = eph.compute_total_group_delay_rnx3(eph_rnx3_df,
                                                   helpers.timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"), "C01",
                                                   "C7D")

    tgd_c2i_s_expected = -5.800000000000E-09
    tgd_c7i_s_expected = -1.020000000000E-08
    tgd_c6i_s_expected = 0

    assert (abs(tgd_c2i_s - tgd_c2i_s_expected) * constants.cGpsIcdSpeedOfLight_mps < 1e-3)
    assert (abs(tgd_c7i_s - tgd_c7i_s_expected) * constants.cGpsIcdSpeedOfLight_mps < 1e-3)
    assert (abs(tgd_c6i_s - tgd_c6i_s_expected) * constants.cGpsIcdSpeedOfLight_mps < 1e-3)
    assert (np.isnan(tgd_c1d_s))
    assert (np.isnan(tgd_c1p_s))
    assert (np.isnan(tgd_c5x_s))
