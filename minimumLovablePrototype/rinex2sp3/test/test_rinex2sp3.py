import math
import pandas as pd
import numpy as np
from pathlib import Path
from ..rinex2sp3 import process
from ..helpers import compressed_2_uncompressed
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


def test_process(input_for_test):
    rinex_nav_file = compressed_2_uncompressed(input_for_test["all_constellations_nav_file"])
    sp3_files = process(rinex_nav_file)
    for sp3_file in sp3_files:
        assert sp3_file.exists(), "SP3 file does not exist"
