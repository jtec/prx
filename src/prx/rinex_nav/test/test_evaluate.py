import pandas as pd
import numpy as np
from pathlib import Path
from .. import evaluate
from  import constants
from ... import constants
from ... import sp3
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
    rx_time = pd.Timestamp("2022-01-01T01:00:00.000000000") - constants.cGpstUtcEpoch
    query_times["G01"] = rx_time + pd.Timedelta(seconds=1) / 1e3
    query_times["E02"] = rx_time + pd.Timedelta(seconds=2) / 1e3
    query_times["C03"] = rx_time + pd.Timedelta(seconds=3) / 1e3
    query_times["R04"] = rx_time + pd.Timedelta(seconds=4) / 1e3
    rinex_sat_states = evaluate.compute(rinex_nav_file, query_times)
    for satellite, query_time in query_times.items():
        sp3_sat_state = sp3.evaluate.compute(input_for_test["sp3_file"], query_time)
        assert np.allclose()
