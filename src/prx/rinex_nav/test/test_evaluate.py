import pandas as pd
import numpy as np
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
            Path(__file__).parent.joinpath(
                "datasets", test_file_path.name
            ),
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
    sat_state_query_time_gpst = pd.Timestamp("2022-01-01T00:2:00.000000000") - constants.cGpstUtcEpoch
    query_times["G01"] = sat_state_query_time_gpst
    query_times["E02"] = sat_state_query_time_gpst
    query_times["C03"] = sat_state_query_time_gpst
    query_times["R04"] = sat_state_query_time_gpst
    sp3_sat_states = sp3_evaluate.compute(input_for_test["sp3_file"], sat_state_query_time_gpst)
    rinex_sat_states = rinex_nav_evaluate.compute(rinex_nav_file, query_times)
    for satellite, query_time in query_times.items():
        sp3_sat_position = sp3_sat_states[sp3_sat_states['sv'] == satellite][['x_m', 'y_m', 'z_m']].to_numpy()
        rinex_sat_position = rinex_sat_states[rinex_sat_states['sv'] == satellite][['x', 'y', 'z']].to_numpy()
        sp3_sat_velocity = sp3_sat_states[sp3_sat_states['sv'] == satellite][['dx_mps', 'dy_mps', 'dz_mps']].to_numpy()
        rinex_sat_velocity = rinex_sat_states[rinex_sat_states['sv'] == satellite][['vx', 'vy', 'vz']].to_numpy()

        # We expect broadcast ephemeris error w.r.t. precise orbits to be less than 10 meters
        assert np.linalg.norm(sp3_sat_position - rinex_sat_position) < 1e1
        # We expect broadcast ephemeris velocity error w.r.t. precise orbits to be less than 1mm/s
        assert np.linalg.norm(sp3_sat_position - rinex_sat_position) < 1e-3

