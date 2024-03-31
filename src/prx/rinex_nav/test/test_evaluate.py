import numpy as np
import pandas as pd
from pathlib import Path
from prx.sp3 import evaluate as sp3_evaluate
from prx.rinex_nav import evaluate as rinex_nav_evaluate
from prx import constants, converters, helpers
from prx.helpers import timestamp_2_timedelta, week_and_seconds_2_timedelta
import shutil
import pytest
import os
import itertools
from dotmap import DotMap

# The following thresholds are the achieved maximum difference between broadcast and
# MGEX precise orbit and clock solutions seen in this test.
# TODO It would be desirable to have an independent reference for expected broadcast values.
# Reasons for differences between SP3 and RINEX:
# - The regular broadcast ephemeris error
# - Different Satellite Reference Points (Center of Mass, iono-free phase center etc.)
# - Using integer-second-aligned GPST (i.e. system time integer-second aligned to GPST) as query time,
#   whereas SP3 uses GPST, so we will see the satellite displacement within time offset between
#   system time and GPST. The offset is typically tens of nanoseconds though, so this should not account
#   for more than millimeter-level, much smaller than the ephemeris error.
expected_max_differences_broadcast_vs_precise = {
    "diff_xyz_l2_m": 11.9,
    "diff_dxyz_l2_mps": 1.8e-4,
    "clock_m": 26,  # TODO Broadcast clock error should be much smaller than this.
    "dclock_mps": 1.2e-4,
}


@pytest.fixture
def input_for_test():
    test_directory = (
        Path(__file__).parent.joinpath(f"./tmp_test_directory_{__name__}").resolve()
    )
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_files = {
        "rinex_nav_file": test_directory / "BRDC00IGS_R_20220010000_01D_MN.zip",
        "sp3_file": test_directory / "WUM0MGXULT_20220010000_01D_05M_ORB.SP3",
    }
    for key, test_file_path in test_files.items():
        shutil.copy(
            Path(__file__).parent.joinpath("datasets", test_file_path.name),
            test_file_path,
        )
        assert test_file_path.exists()
    yield test_files
    shutil.rmtree(test_directory)


def test_compare_rnx3_gps_sat_pos_with_magnitude(input_for_test):
    """Loads a RNX3 nav file, computes broadcast position for a GPS satellite and compares to
    position computed by MAGNITUDE matlab library"""
    path_to_rnx3_nav_file = converters.anything_to_rinex_3(
        input_for_test["rinex_nav_file"]
    )
    query = pd.DataFrame(
        {
            "sv": "G01",
            "query_time_isagpst": week_and_seconds_2_timedelta(
                weeks=2190, seconds=523800
            ),
        },
        index=[0],
    )
    rinex_sat_states = rinex_nav_evaluate.compute(path_to_rnx3_nav_file, query)

    # MAGNITUDE position
    sv_pos_magnitude = np.array([13053451.235, -12567273.060, 19015357.126])
    sv_pos_prx = rinex_sat_states[["x_m", "y_m", "z_m"]][
        rinex_sat_states.sv == "G01"
        ].to_numpy()

    threshold_pos_error_m = 1e-3
    assert np.linalg.norm(sv_pos_prx - sv_pos_magnitude) < threshold_pos_error_m


def generate_sat_query(sat_state_query_time_isagpst):
    query = pd.DataFrame(
        [
            # Multiple satellites with ephemerides provided as Kepler orbits
            # Two Beidou GEO (from http://www.csno-tarc.cn/en/system/constellation)
            {
                "sv": "C03",
                "signal": "C2I",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            {
                "sv": "C05",
                "signal": "C2I",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            # One Beidou IGSO
            {
                "sv": "C38",
                "signal": "C2I",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            # One Beidou MEO
            {
                "sv": "C30",
                "signal": "C2I",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            # Two GPS
            {
                "sv": "G15",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            {
                "sv": "G12",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            # Query one GPS satellite at two different times to cover that case
            {
                "sv": "G15",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst
                                      + pd.Timedelta(seconds=1),
            },
            # Two Galileo
            {
                "sv": "E24",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            {
                "sv": "E30",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            # Two QZSS
            {
                "sv": "J02",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            {
                "sv": "J03",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            # Multiple satellites with orbits that require propagation of an initial state
            # Two GLONASS satellites
            {
                "sv": "R04",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
            {
                "sv": "R05",
                "signal": "C1C",
                "query_time_isagpst": sat_state_query_time_isagpst,
            },
        ]
    )
    return query


def test_compare_to_sp3(input_for_test):
    rinex_nav_file = converters.compressed_to_uncompressed(
        input_for_test["rinex_nav_file"]
    )
    query = generate_sat_query(pd.Timestamp("2022-01-01T01:10:00.000000000"))

    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query.copy())
    rinex_sat_states = (
        rinex_sat_states.sort_values(by=["sv", "query_time_isagpst"])
        .sort_index(axis=1)
        .reset_index()
        .drop(columns=["index", "signal", "group_delay_m"])
    )

    sp3_sat_states = sp3_evaluate.compute(
        input_for_test["sp3_file"], query.copy().drop(columns=["signal"])
    )
    sp3_sat_states = (
        sp3_sat_states.sort_values(by=["sv", "query_time_isagpst"])
        .sort_index(axis=1)
        .reset_index()
        .drop(columns=["index"])
    )

    # Verify that the SP3 states are ordered the same as the RINEX states
    assert sp3_sat_states["sv"].equals(rinex_sat_states["sv"])
    # Verify that query times are the same for each row. We can check for equality here
    # as the timestamps are based on integers with nanosecond resolution, so no need to
    # worry about floating point precision.
    assert sp3_sat_states["query_time_isagpst"].equals(
        rinex_sat_states["query_time_isagpst"]
    )
    # Verify that sorting columns worked as expected
    assert sp3_sat_states.columns.equals(rinex_sat_states.columns)
    diff = rinex_sat_states.drop(columns="sv") - sp3_sat_states.drop(columns="sv")
    diff = pd.concat((rinex_sat_states["sv"], diff), axis=1)
    diff["diff_xyz_l2_m"] = np.linalg.norm(
        diff[["x_m", "y_m", "z_m"]].to_numpy(), axis=1
    )
    diff["diff_dxyz_l2_mps"] = np.linalg.norm(
        diff[["dx_mps", "dy_mps", "dz_mps"]].to_numpy(), axis=1
    )

    print("\n" + diff.to_string())
    for (
            column,
            expected_max_difference,
    ) in expected_max_differences_broadcast_vs_precise.items():
        assert not diff[column].isnull().values.any()
        assert (
                diff[column].max() < expected_max_difference
        ), f"Expected maximum difference {expected_max_difference} for column {column}, but got {diff[column].max()}"


@pytest.fixture
def set_up_test_2023():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_files = DotMap()
    test_files.nav_file = test_directory.joinpath(
        "BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
    )
    test_files.sp3_file = test_directory.joinpath(
        "GFZ0MGXRAP_20230010000_01D_05M_ORB.SP3"
    )

    for key, test_file in test_files.items():
        shutil.copy(
            helpers.prx_repository_root()
            / f"src/prx/test/datasets/TLSE_2023001/{test_file.name}",
            test_file,
        )
        assert test_file.exists()

    yield dict(test_files)
    shutil.rmtree(test_directory)


def test_2023_beidou_c27(set_up_test_2023):
    rinex_nav_file = converters.compressed_to_uncompressed(set_up_test_2023["nav_file"])
    query = pd.DataFrame(
        [
            # Multiple satellites with ephemerides provided as Kepler orbits
            # Two Beidou GEO (from http://www.csno-tarc.cn/en/system/constellation)
            {
                "sv": "C27",
                "signal": "C1X",
                "query_time_isagpst": pd.Timestamp("2023-01-01T01:00:00.000000000")
                                      - constants.cGpstUtcEpoch,
            },
        ]
    )

    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query.copy())
    assert (
            len(rinex_sat_states) == 1
    ), "Was expecting only one, row, make sure to sort before comparing to sp3 with more than one row"
    rinex_sat_states = (
        rinex_sat_states.reset_index()
        .drop(columns=["index", "signal", "group_delay_m"])
        .sort_index(axis="columns")
    )
    rinex_sat_states.to_csv("jan.csv")
    sp3_sat_states = (
        sp3_evaluate.compute(set_up_test_2023["sp3_file"], query.copy())
        .drop(columns=["signal"])
        .sort_index(axis="columns")
    )
    assert sp3_sat_states.columns.equals(rinex_sat_states.columns)
    diff = rinex_sat_states.drop(columns="sv") - sp3_sat_states.drop(columns="sv")
    diff = pd.concat((rinex_sat_states["sv"], diff), axis=1)
    diff["diff_xyz_l2_m"] = np.linalg.norm(
        diff[["x_m", "y_m", "z_m"]].to_numpy(), axis=1
    )
    diff["diff_dxyz_l2_mps"] = np.linalg.norm(
        diff[["dx_mps", "dy_mps", "dz_mps"]].to_numpy(), axis=1
    )
    for (
            column,
            expected_max_difference,
    ) in expected_max_differences_broadcast_vs_precise.items():
        assert (
                diff[column].max() < expected_max_difference
        ), f"Expected maximum difference {expected_max_difference} for column {column}, but got {diff[column].max()}"


def test_group_delays(input_for_test):
    rinex_nav_file = converters.compressed_to_uncompressed(
        input_for_test["rinex_nav_file"]
    )
    query_time_isagpst = (
            pd.Timestamp("2022-01-01T01:10:00.000000000") - constants.cGpstUtcEpoch
    )
    query = pd.DataFrame(
        [
            {"sv": "C30", "signal": "C2I", "query_time_isagpst": query_time_isagpst},
            {"sv": "C30", "signal": "C7I", "query_time_isagpst": query_time_isagpst},
            {"sv": "G15", "signal": "C1C", "query_time_isagpst": query_time_isagpst},
            # Query one GPS satellite at two different times to cover that case
            {
                "sv": "G15",
                "signal": "C2C",
                "query_time_isagpst": query_time_isagpst + pd.Timedelta(seconds=1),
            },
            {"sv": "E24", "signal": "C1A", "query_time_isagpst": query_time_isagpst},
            {"sv": "E24", "signal": "C5Q", "query_time_isagpst": query_time_isagpst},
            {"sv": "J02", "signal": "C1C", "query_time_isagpst": query_time_isagpst},
            {"sv": "J02", "signal": "C2S", "query_time_isagpst": query_time_isagpst},
        ]
    )
    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query)
    # Check that group delays are computed for all signals
    assert not rinex_sat_states["group_delay_m"].isna().any()


def max_abs_diff_smaller_than(a, b, threshold):
    if isinstance(a, pd.Series):
        a = a.to_numpy()
    if isinstance(b, pd.Series):
        b = b.to_numpy()
    return np.max(np.abs(a - b)) < threshold


def test_gps_group_delay(input_for_test):
    """
    Computes the total group delay (TGD) for GPS from a RNX3 NAV file containing the ephemerides pasted below.
    The RINEX navigation message field containing TGD is highlighted between **

    This tests also validates
    - the choice of the right ephemeris for the correct time: 3 epochs are used
    - the scaling of the TGD with the carrier frequency: the 3 observations types considered in IS-GPS-200N are tested (
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
    rinex_3_navigation_file = converters.anything_to_rinex_3(
        input_for_test["rinex_nav_file"]
    )
    # Retrieve total group delays for 4 different observation codes, at 3 different times
    codes = ["C1C", "C1P", "C2P", "C5X"]
    times = [
        timestamp_2_timedelta(pd.Timestamp("2022-01-01T00:00:00.000000000"), "GPST"),
        timestamp_2_timedelta(pd.Timestamp("2022-01-01T01:30:00.000000000"), "GPST"),
        timestamp_2_timedelta(pd.Timestamp("2022-01-01T02:15:00.000000000"), "GPST"),
    ]
    query = pd.DataFrame()
    for code, time in itertools.product(codes, times):
        query = pd.concat(
            [
                query,
                pd.DataFrame(
                    [{"sv": "G02", "signal": code, "query_time_isagpst": time}]
                ),
            ]
        )
    tgds = rinex_nav_evaluate.compute(rinex_3_navigation_file, query)
    # Verify that rows are in chronological order
    for code in codes:
        assert (
            tgds[tgds.signal == code]["query_time_isagpst"]
            .reset_index(drop=True)
            .equals(pd.Series(times))
        )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C1C"]["group_delay_m"],
        constants.cGpsSpeedOfLight_mps
        * pd.Series([-1.769512891770e-08, -1.769512891770e-08, -1.769512891769e-08]),
        1e-6,
    )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C1P"]["group_delay_m"],
        constants.cGpsSpeedOfLight_mps
        * pd.Series([-1.769512891770e-08, -1.769512891770e-08, -1.769512891769e-08]),
        1e-6,
    )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C2P"]["group_delay_m"],
        constants.cGpsSpeedOfLight_mps
        * pd.Series([-1.769512891770e-08, -1.769512891770e-08, -1.769512891769e-08])
        * (
                constants.carrier_frequencies_hz()["G"]["L1"]
                / constants.carrier_frequencies_hz()["G"]["L2"]
        )
        ** 2,
        1e-6,
    )
    assert np.all(np.isnan(tgds[tgds.signal == "C5X"]["group_delay_m"].to_numpy()))


def test_gal_group_delay(input_for_test):
    """
    Note that both ephemerides have the same Toe (reference time), but one is I/NAV, the
    other F/NAV. This test implicitly checks whether the right ephemeris is used wben computing
    TGDs.

    E25 2022 01 01 00 00 00-5.587508785538e-04-1.278976924368e-12 0.000000000000e+00
         9.600000000000e+01 1.960625000000e+02 2.576178736756e-09 1.034878564309e+00
         9.007751941681e-06 3.147422103211e-04 1.221708953381e-05 5.440630062103e+03
         5.184000000000e+05 1.303851604462e-08 2.996653673305e+00 9.313225746155e-09
         9.752953840989e-01 8.875000000000e+01-5.349579038983e-01-5.132356640398e-09
        -2.260808457461e-10 2.580000000000e+02 2.190000000000e+03 0.000000000000e+00
         3.120000000000e+00 0.000000000000e+00 **4.423782229424e-09** **0.000000000000e+00**
         5.191430000000e+05 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00
    E25 2022 01 01 00 00 00-5.587504128930e-04-1.278976924370e-12 0.000000000000e+00
         9.600000000000e+01 1.960625000000e+02 2.576178736760e-09 1.034878564310e+00
         9.007751941680e-06 3.147422103210e-04 1.221708953380e-05 5.440630062100e+03
         5.184000000000e+05 1.303851604460e-08 2.996653673310e+00 9.313225746150e-09
         9.752953840990e-01 8.875000000000e+01-5.349579038980e-01-5.132356640400e-09
        -2.260808457460e-10 5.160000000000e+02 2.190000000000e+03
         3.120000000000e+00 0.000000000000e+00 **4.423782229420e-09** **4.889443516730e-09**
         5.190640000000e+05
    """
    rinex_3_navigation_file = converters.anything_to_rinex_3(
        input_for_test["rinex_nav_file"]
    )
    query = pd.DataFrame()
    codes = ["C1C", "C5X", "C7X", "C6B"]
    for code in codes:
        code_query = pd.DataFrame(
            [
                {
                    "sv": "E25",
                    "signal": code,
                    "query_time_isagpst": rinex_nav_evaluate.to_isagpst(
                        timestamp_2_timedelta(
                            pd.Timestamp("2022-01-01T01:30:00.000000000"), "GST"
                        ),
                        "GST",
                    ),
                }
            ]
        )
        query = pd.concat(
            [
                query,
                code_query,
            ]
        )
    tgds = rinex_nav_evaluate.compute(rinex_3_navigation_file, query)
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C1C"]["group_delay_m"],
        4.889443516730e-09 * constants.cGpsSpeedOfLight_mps,
        1e-6,
    )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C5X"]["group_delay_m"],
        4.423782229424e-09
        * (
                constants.carrier_frequencies_hz()["E"]["L1"]
                / constants.carrier_frequencies_hz()["E"]["L5"]
        )
        ** 2
        * constants.cGpsSpeedOfLight_mps,
        1e-6,
    )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C7X"]["group_delay_m"],
        4.889443516730e-09
        * (
                constants.carrier_frequencies_hz()["E"]["L1"]
                / constants.carrier_frequencies_hz()["E"]["L7"]
        )
        ** 2
        * constants.cGpsSpeedOfLight_mps,
        1e-6,
    )
    assert np.all(np.isnan(tgds[tgds.signal == "C6B"]["group_delay_m"].to_numpy()))


def test_bds_group_delay(input_for_test):
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
    rinex_3_navigation_file = converters.anything_to_rinex_3(
        input_for_test["rinex_nav_file"]
    )
    query = pd.DataFrame()
    codes = [
        "C2I",  # B1I -> C2I
        "C7I",  # B2I -> C7I
        "C6I",  # B3I -> C6I
        "C1D",  # B1Cd -> C1D
        "C1P",  # B1Cp -> C1P
        "C7D",  # B2bi -> C7D
    ]
    for code in codes:
        code_query = pd.DataFrame(
            {
                "sv": "C01",
                "signal": code,
                "query_time_isagpst": rinex_nav_evaluate.to_isagpst(
                    timestamp_2_timedelta(
                        pd.Timestamp("2022-01-01T00:30:00.000000000"), "BDT"
                    ),
                    "BDT",
                ),
            },
            index=[0],
        )
        query = pd.concat(
            [
                query,
                code_query,
            ]
        )
    tgds = rinex_nav_evaluate.compute(rinex_3_navigation_file, query)

    tgd_c2i_s_expected = -5.800000000000e-09 * constants.cGpsSpeedOfLight_mps
    tgd_c7i_s_expected = -1.020000000000e-08 * constants.cGpsSpeedOfLight_mps
    tgd_c6i_s_expected = 0 * constants.cGpsSpeedOfLight_mps

    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C2I"].group_delay_m, tgd_c2i_s_expected, 1e-6
    )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C7I"].group_delay_m, tgd_c7i_s_expected, 1e-6
    )
    assert max_abs_diff_smaller_than(
        tgds[tgds.signal == "C6I"].group_delay_m, tgd_c6i_s_expected, 1e-6
    )
    assert np.all(np.isnan(tgds[tgds.signal == "C1D"]["group_delay_m"].to_numpy()))
    assert np.all(np.isnan(tgds[tgds.signal == "C1P"]["group_delay_m"].to_numpy()))
    assert np.all(np.isnan(tgds[tgds.signal == "C5X"]["group_delay_m"].to_numpy()))
