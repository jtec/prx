import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd
import pytest

from prx import helpers
from prx import constants
from prx import main


# This function sets up a temporary directory, copies a rinex observations file into that directory
# and returns its path. The @pytest.fixture annotation allows us to pass the function as an input
# to test functions. When running a test function, pytest will then first run this function, pass
# whatever is passed to `yield` to the test function, and run the code after `yield` after the test,
# even  if the test crashes.
@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    test_file = test_directory.joinpath(compressed_compact_rinex_file)
    shutil.copy(
        Path(__file__).parent
        / f"datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_file,
    )
    assert test_file.exists()
    # Also provide ephemerides so the test does not have to download them:
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{ephemerides_file}",
        test_file.parent.joinpath(ephemerides_file),
    )
    assert test_file.parent.joinpath(ephemerides_file).exists()

    yield test_file
    # shutil.rmtree(test_file.parent)


@pytest.fixture
def input_for_test_with_first_epoch_at_midnight():
    # Having a first epoch at midnight requires to have the NAV data from the previous day, because we are computing
    # the time of emission as (time of reception - pseudorange/celerity)
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    # filepath_to_mixed_obs_file = "TLSE00FRA_R_20220010000_01H_30S_MO.rnx.gz"
    # filepath_to_mixed_obs_file = "TLSE00FRA_R_20220010000_01H_30S_GO.zip"
    # test_mixed_obs_file = test_directory.joinpath(filepath_to_mixed_obs_file)
    # shutil.copy(
    #     Path(__file__).parent/f"datasets/TLSE_2022001/{filepath_to_mixed_obs_file}",
    #     test_mixed_obs_file,
    # )
    filepath_to_mixed_obs_file = "TLSE00FRA_R_20230010000_30M_30S_GO.crx.gz"
    test_mixed_obs_file = test_directory.joinpath(filepath_to_mixed_obs_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{filepath_to_mixed_obs_file}",
        test_mixed_obs_file,
    )
    assert test_mixed_obs_file.exists()

    # nav data from same day
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/BRDC00IGS_R_20230010000_01D_MN.rnx.zip",
        test_directory.joinpath("BRDC00IGS_R_20230010000_01D_MN.rnx.zip"),
    )
    assert test_mixed_obs_file.exists()
    # nav data from previous day
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/BRDC00IGS_R_20223650000_01D_MN.rnx.gz",
        test_directory.joinpath("BRDC00IGS_R_20223650000_01D_MN.rnx.gz"),
    )

    yield {"mixed_obs_file": test_mixed_obs_file,}
    shutil.rmtree(test_directory)


def test_prx_command_line_call_with_csv_output(input_for_test):
    test_file = input_for_test
    prx_path = helpers.prx_repository_root() / "src/prx/main.py"
    command = (
        f"python {prx_path} --observation_file_path {test_file} --output_format csv"
    )
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert result.returncode == 0
    assert expected_prx_file.exists()


def test_prx_function_call_with_csv_output(input_for_test):
    test_file = input_for_test
    main.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()
    df = pd.read_csv(expected_prx_file, comment="#")
    assert not df.empty
    # Elevation sanity check
    assert (
        df[(df.prn == 14) & (df.constellation == "C")].elevation_deg - 34.86
    ).abs().max() < 0.3


def test_prx_function_call_for_obs_file_across_two_days(input_for_test_with_first_epoch_at_midnight):
    test_file = input_for_test_with_first_epoch_at_midnight["mixed_obs_file"]
    main.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()