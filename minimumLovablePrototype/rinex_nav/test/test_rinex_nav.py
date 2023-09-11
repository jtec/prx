import math
import pandas as pd
import numpy as np
from pathlib import Path
from .. import evaluate
from ..helpers import compressed_2_uncompressed
from ..constants import cGpstUtcEpoch
import shutil
import pytest
import os


@pytest.fixture
def input_for_test():
    test_directory = Path(__file__).parent.joinpath(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    all_constellations_rnx3_nav_test_file = test_directory.joinpath(
        "BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    shutil.copy(Path(__file__).parent.joinpath("datasets", "BRDC00IGS_R_20220010000_01D_MN.zip"), all_constellations_rnx3_nav_test_file)
    assert all_constellations_rnx3_nav_test_file.exists()
    yield {
        "all_constellations_nav_file": all_constellations_rnx3_nav_test_file,
    }

    shutil.rmtree(test_directory)


def test_position(input_for_test):
    rinex_nav_file = compressed_2_uncompressed(input_for_test["all_constellations_nav_file"])
    query_times = {}
    rx_time = pd.Timestamp(np.datetime64("2022-01-01T01:00:00.000000000")) - cGpstUtcEpoch
    query_times['G01'] = rx_time + pd.Timedelta(seconds=1) / 1e3
    query_times['E02'] = rx_time + pd.Timedelta(seconds=2) / 1e3
    query_times['C03'] = rx_time + pd.Timedelta(seconds=3) / 1e3
    query_times['R04'] = rx_time + pd.Timedelta(seconds=4) / 1e3
    sat_states = evaluate.compute(rinex_nav_file, query_times)
    pass
