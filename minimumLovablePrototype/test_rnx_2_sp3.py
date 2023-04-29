import rinex_2_sp3
import pytest
from pathlib import Path
import os
import shutil
import helpers


# This function sets up a temporary directory, copies a zipped rinex navigation file into that directory
# and returns its path. Tests in this file will then operate on that file.
# The function also deletes the temporary directory after the test is run.
#
# Thanks to the @pytest.fixtures annotation, when running a test function, pytest will
# first run this function, pass whatever is passed to `yield` to the test function, and
# run the code after `yield` after the test, even  if the test crashes.
@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    all_constellations_rnx3_nav_test_file = test_directory.joinpath(
        "BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    shutil.copy(
        helpers.prx_root().joinpath(
            f"datasets/TLSE_2022001/{all_constellations_rnx3_nav_test_file.name}"
        ),
        all_constellations_rnx3_nav_test_file,
    )
    assert all_constellations_rnx3_nav_test_file.exists()

    yield {
        "all_constellations_nav_file": all_constellations_rnx3_nav_test_file,
    }
    shutil.rmtree(test_directory)


def test_that_sp3_files_are_generated(input_for_test):
    assert input_for_test["all_constellations_nav_file"].exists()
    sp3_files = rinex_2_sp3.generate_sp3(input_for_test["all_constellations_nav_file"])
    for sp3_file in sp3_files:
        assert sp3_file.exists()