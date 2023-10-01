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
    test_directory = (
        Path(__file__).parent.joinpath(f"./tmp_test_directory_{__name__}").resolve()
    )
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    test_file_one_sample_removed = test_directory.joinpath("WUM0MGXULT_20220010000_01H_05M_ORB_one_sample_removed.SP3")
    shutil.copy(Path(__file__).parent.joinpath("datasets", test_file_one_sample_removed.name), test_file_one_sample_removed)

    assert test_file_one_sample_removed.exists()
    yield {
        "test_file_one_sample_removed": test_file_one_sample_removed,

    }
    shutil.rmtree(test_directory)


def test_at_sample(input_for_test):
    sp3_file = input_for_test["test_file_one_sample_removed"]
    # Compute satellite states directly at a sample time somewhere in the middle of the file
    query_time = pd.Timestamp("2021-12-31T00:20:00.00000000") - constants.cGpstUtcEpoch
    sat_states = compute(sp3_file, query_time)
    assert np.allclose(sat_states[sat_states["sv"] == "G01"][["x_m", "y_m", "z_m"]].to_numpy(),
        1e3*np.array([13744.907145, 20823.122313,  8309.113118]), rtol=1e-5, atol=1e-3)
    assert np.allclose(sat_states[sat_states["sv"] == "G01"][["clock_s"]].to_numpy(),
        np.array([469.979467 / constants.cMicrosecondsPerSecond]),
                       rtol=1e-5, atol=1e-3 / constants.cGpsIcdSpeedOfLight_mps)


def test_between_samples(input_for_test):
    sp3_file = input_for_test["test_file_one_sample_removed"]
    # Compute satellite states directly at a sample time somewhere in the middle of the file
    query_time = pd.Timestamp("2021-12-31T00:30:00.00000000") - constants.cGpstUtcEpoch
    sat_states = compute(sp3_file, query_time)
    assert sat_states[sat_states["sv"] == "G01"][["x_m", "y_m", "z_m"]].to_numpy()[0] == pytest.approx(
        1e3*np.array([13624.009028, -20092.399598,  10082.111937]), abs=1e-3
    )
    # Note that the tolerance has been relaxed to a centimeter here to accommodate interpolation error
    assert sat_states[sat_states["sv"] == "G01"][["clock_s"]].values[
        0
    ] == pytest.approx(
        469.973744 / constants.cMicrosecondsPerSecond,
        abs=1e-2 / constants.cGpsIcdSpeedOfLight_mps,
    )