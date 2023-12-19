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
        {"satellite": "C03", "query_time_isagpst": sat_state_query_time_isagpst},
        {"satellite": "C05", "query_time_isagpst": sat_state_query_time_isagpst},
        # One Beidou IGSO
    {"satellite": "C38", "query_time_isagpst": sat_state_query_time_isagpst},
    # One Beidou MEO
    {"satellite": "C30", "query_time_isagpst": sat_state_query_time_isagpst},
    # Two GPS
    {"satellite": "G15", "query_time_isagpst": sat_state_query_time_isagpst},
    {"satellite": "G12", "query_time_isagpst": sat_state_query_time_isagpst},
    # Two Galileo
    {"satellite": "E24", "query_time_isagpst": sat_state_query_time_isagpst},
    {"satellite": "E30", "query_time_isagpst": sat_state_query_time_isagpst},
    # Two QZSS
    {"satellite": "J02", "query_time_isagpst": sat_state_query_time_isagpst},
    {"satellite": "J03", "query_time_isagpst": sat_state_query_time_isagpst},
    # Multiple satellites with orbits that require propagation of an initial state
    # Two GLONASS satellites
    # {"satellite": "R04", "query_time_isagpst": sat_state_query_time_isagpst},
    # {"satellite": "R05", "query_time_isagpst": sat_state_query_time_isagpst},
    ])

    sp3_sat_states = sp3_evaluate.compute(
        input_for_test["sp3_file"], query
    )
    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query)
    for satellite, query_time in query_times.items():
        sp3 = {
            "position_m": sp3_sat_states[sp3_sat_states["sv"] == satellite][
                ["x_m", "y_m", "z_m"]
            ].to_numpy(),
            "velocity_mps": sp3_sat_states[sp3_sat_states["sv"] == satellite][
                ["dx_mps", "dy_mps", "dz_mps"]
            ].to_numpy(),
            "clock_m": sp3_sat_states[sp3_sat_states["sv"] == satellite][
                ["clock_m"]
            ].to_numpy(),
            "dclock_mps": sp3_sat_states[sp3_sat_states["sv"] == satellite][
                ["dclock_mps"]
            ].to_numpy(),
        }
        rinex = {
            "position_m": rinex_sat_states[rinex_sat_states["sv"] == satellite][
                ["x_m", "y_m", "z_m"]
            ].to_numpy(),
            "velocity_mps": rinex_sat_states[rinex_sat_states["sv"] == satellite][
                ["vx_mps", "vy_mps", "vz_mps"]
            ].to_numpy(),
            "clock_m": rinex_sat_states[rinex_sat_states["sv"] == satellite][
                ["clock_offset_m"]
            ].to_numpy(),
            "dclock_mps": rinex_sat_states[rinex_sat_states["sv"] == satellite][
                ["clock_offset_rate_mps"]
            ].to_numpy(),
        }
        # These thresholds are based on the achieved maximum difference between broadcast and
        # MGEX precise orbit and clock solutions in this test.
        expected_max_difference_l2 = {
            "position_m": 11.9,
            "velocity_mps": 2e-4,
            "clock_m": 26,
            "dclock_mps": 2e-4,
        }
        max_diffs_l2norm = {}
        for state_name in sp3:
            diff = rinex[state_name] - sp3[state_name]
            assert np.linalg.norm(diff, 2) < expected_max_difference_l2[state_name]
            print(
                f"\n satellite: {satellite}, state: {state_name}, diff: {diff} (norm: {np.linalg.norm(diff)}) [m or m/s] (sp3: {sp3[state_name]}, rinex: {rinex[state_name]}))"
            )
            if state_name not in max_diffs_l2norm or np.linalg.norm(
                diff, 2
            ) > np.linalg.norm(max_diffs_l2norm[state_name]["difference"], 2):
                max_diffs_l2norm[state_name] = {"difference": diff, "satellite": satellite}
    print("Maximum differences between broadcast and MGEX precise solutions:")
    for state_name, max_diff in max_diffs_l2norm.items():
        print(
            f"\n state: {state_name}, satellite: {max_diff['satellite']}, diff: {max_diff['difference']} (l2 norm: {np.linalg.norm(max_diff['difference'], 2)}) [m or m/s])"
        )
