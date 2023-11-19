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


def test_position(input_for_test):
    rinex_nav_file = converters.compressed_to_uncompressed(
        input_for_test["rinex_nav_file"]
    )
    query_times = {}
    sat_state_query_time_gpst = (
        pd.Timestamp("2022-01-01T01:10:00.000000000") - constants.cGpstUtcEpoch
    )
    # Time-of-transmission is different for each satellite, simulate that here
    # Multiple satellites with ephemerides provided as Kepler orbits
    # Two Beidou GEO (from http://www.csno-tarc.cn/en/system/constellation)
    query_times["C03"] = sat_state_query_time_gpst
    # query_times["C05"] = sat_state_query_time_gpst
    # One Beidou IGSO
    # query_times["C38"] = sat_state_query_time_gpst
    # One Beidou MEO
    # query_times["C30"] = sat_state_query_time_gpst
    # Two GPS
    # query_times["G15"] = sat_state_query_time_gpst
    # query_times["G12"] = sat_state_query_time_gpst
    # Two Galileo
    # query_times["E24"] = sat_state_query_time_gpst
    # query_times["E30"] = sat_state_query_time_gpst
    # Two QZSS
    # query_times["J02"] = sat_state_query_time_gpst
    # query_times["J03"] = sat_state_query_time_gpst

    # Multiple satellites with orbits that require propagation of an initial state
    # Two GLONASS satellites
    # query_times["R04"] = sat_state_query_time_gpst + pd.Timedelta(seconds=20)/1e3
    # query_times["R05"] = sat_state_query_time_gpst + pd.Timedelta(seconds=21)/1e3
    sp3_sat_states = sp3_evaluate.compute(
        input_for_test["sp3_file"], sat_state_query_time_gpst
    )
    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query_times)
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
        # These thresholds are based on the expected maximum difference between broadcast and
        # MGEX precise orbit and clock solutions.
        expected_max_abs_difference = {
            "position_m": 2,
            "velocity_mps": 1e-3,
            "clock_m": 22,
            "dclock_mps": 3e-3,
        }
        for state_name in sp3:
            diff = rinex[state_name] - sp3[state_name]
            print(
                f"\n satellite: {satellite}, state: {state_name}, diff: {diff} [m or m/s] (sp3: {sp3[state_name]}, rinex: {rinex[state_name]}))"
            )
            # assert np.linalg.norm(diff) < expected_max_abs_difference[state_name]
            pass
