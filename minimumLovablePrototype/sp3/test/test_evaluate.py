import math
import pandas as pd
import numpy as np
from pathlib import Path
from ..evaluate import compute
import shutil
import pytest
import os
from ... import constants


@pytest.fixture
def input_for_test():
    test_directory = Path(__file__).parent.joinpath(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    test_file = test_directory.joinpath(
        "WUM0MGXULT_20220010000_01D_05M_ORB.SP3"
    )
    shutil.copy(Path(__file__).parent.joinpath("datasets", test_file.name), test_file)
    assert test_file.exists()
    yield {
        "test_file": test_file,
    }

    shutil.rmtree(test_directory)


def test_position(input_for_test):
    sp3_file = input_for_test["test_file"]
    # Compute satellite states directly at a sample time somewhere in the middle of the file
    query_time = pd.Timestamp("2022-01-01T02:10:00.00000000") - constants.cGpstUtcEpoch
    sat_states = compute(sp3_file, query_time)
    pass
