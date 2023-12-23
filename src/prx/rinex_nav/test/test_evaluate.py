import numpy as np
import pandas as pd
from pathlib import Path
from prx.sp3 import evaluate as sp3_evaluate
from prx.rinex_nav import evaluate as rinex_nav_evaluate
from prx import converters
from prx import constants
import shutil
import pytest
import os


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


def test_compare_to_sp3(input_for_test):
    rinex_nav_file = converters.compressed_to_uncompressed(
        input_for_test["rinex_nav_file"]
    )
    query_times = {}
    sat_state_query_time_isagpst = (
        pd.Timestamp("2022-01-01T01:10:00.000000000") - constants.cGpstUtcEpoch
    )
    query = pd.DataFrame([
    # Multiple satellites with ephemerides provided as Kepler orbits
    # Two Beidou GEO (from http://www.csno-tarc.cn/en/system/constellation)
        {"sv": "C03", "query_time_isagpst": sat_state_query_time_isagpst},
        {"sv": "C05", "query_time_isagpst": sat_state_query_time_isagpst},
        # One Beidou IGSO
    {"sv": "C38", "query_time_isagpst": sat_state_query_time_isagpst},
    # One Beidou MEO
    {"sv": "C30", "query_time_isagpst": sat_state_query_time_isagpst},
    # Two GPS
    {"sv": "G15", "query_time_isagpst": sat_state_query_time_isagpst},
    {"sv": "G12", "query_time_isagpst": sat_state_query_time_isagpst},
    # Query one GPS satellite at two different times to cover that case
    {"sv": "G15", "query_time_isagpst": sat_state_query_time_isagpst + pd.Timedelta(seconds=1)},
    # Two Galileo
    {"sv": "E24", "query_time_isagpst": sat_state_query_time_isagpst},
    {"sv": "E30", "query_time_isagpst": sat_state_query_time_isagpst},
    # Two QZSS
    {"sv": "J02", "query_time_isagpst": sat_state_query_time_isagpst},
    {"sv": "J03", "query_time_isagpst": sat_state_query_time_isagpst},
    # Multiple satellites with orbits that require propagation of an initial state
    # Two GLONASS satellites
    # {"sv": "R04", "query_time_isagpst": sat_state_query_time_isagpst},
    # {"sv": "R05", "query_time_isagpst": sat_state_query_time_isagpst},
    ])

    sp3_sat_states = sp3_evaluate.compute(
        input_for_test["sp3_file"], query
    )
    sp3_sat_states = sp3_sat_states.sort_values(by=["sv", "query_time_isagpst"]).sort_index(axis=1).reset_index().drop(columns=["index"])
    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query)
    rinex_sat_states = rinex_sat_states.sort_values(by=["sv", "query_time_isagpst"]).sort_index(axis=1).reset_index().drop(columns=["index"])
    # Verify that the SP3 states are ordered the same as the RINEX states
    assert sp3_sat_states["sv"].equals(rinex_sat_states["sv"])
    # Verify that query times are the same for each row. We can check for equality here
    # as the timestamps are based on integers with nanosecond resolution, so no need to
    # worry about floating point precision.
    assert sp3_sat_states["query_time_isagpst"].equals(rinex_sat_states["query_time_isagpst"])
    # Verify that sorting columns worked as expected
    assert sp3_sat_states.columns.equals(rinex_sat_states.columns)
    diff = rinex_sat_states.drop(columns='sv') - sp3_sat_states.drop(columns='sv')
    diff = pd.concat((rinex_sat_states['sv'], diff), axis=1)
    diff['diff_xyz_l2_m'] = np.linalg.norm(diff[['x_m', 'y_m', 'z_m']].to_numpy(), axis=1)
    diff['diff_dxyz_l2_mps'] = np.linalg.norm(diff[['dx_mps', 'dy_mps', 'dz_mps']].to_numpy(), axis=1)

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
    expected_max_differences = {
        "diff_xyz_l2_m": 11.9,
        "diff_dxyz_l2_mps": 1.8e-4,
        "clock_m": 26,  # TODO Broadcast clock error should be much smaller than this.
        "dclock_mps": 1.2e-4,
    }
    print('\n' + diff.to_string())
    for column, expected_max_difference in expected_max_differences.items():
        assert diff[column].max() < expected_max_difference, f"Expected maximum difference {expected_max_difference} for column {column}, but got {diff[column].max()}"