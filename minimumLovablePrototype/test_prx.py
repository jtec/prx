import os
from pathlib import Path
import shutil
import subprocess

import pytest

import prx
import helpers
import constants

def copy_data_file_to_test_directory(filepath,test_directory):
    filename = filepath.split("/")[-1]
    test_file = test_directory.joinpath(filename)
    shutil.copy(
        helpers.prx_root().joinpath(filepath),
        test_file,
    )
    assert test_file.exists()
    return test_file

def create_test_directory():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    return test_directory

# This function sets up a temporary directory, copies a rinex observations file into that directory
# and returns its path. The @pytest.fixture annotation allows us to pass the function as an input
# to test functions. When running a test function, pytest will then first run this function, pass
# whatever is passed to `yield` to the test function, and run the code after `yield` after the test,
# even  if the test crashes.
@pytest.fixture
def input_for_test():
    test_directory = create_test_directory()

    compressed_compact_rinex_filepath = "datasets/TLSE_2023001//TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    test_file = copy_data_file_to_test_directory(compressed_compact_rinex_filepath,
                                                 test_directory, )

    copy_data_file_to_test_directory("datasets/TLSE_2023001/BRDC00IGS_R_20230010000_01D_MN.rnx.zip",
                                     test_directory)

    yield test_file
    shutil.rmtree(test_file.parent)

@pytest.fixture
def input_for_test_with_first_epoch_at_midnight():
    # Having a first epoch at midnight requires to have the NAV data from the previous day, because we are computing
    # the time of emission as (time of reception - pseudorange/celerity)
    test_directory = create_test_directory()

    filepath_to_mixed_obs_file = "datasets/TLSE_2022001/TLSE00FRA_R_20220010000_01H_30S_MO.rnx.gz"
    test_mixed_obs_file = copy_data_file_to_test_directory(filepath_to_mixed_obs_file,
                                                           test_directory,)
    # nav data from same day
    copy_data_file_to_test_directory("datasets/TLSE_2022001/BRDC00IGS_R_20220010000_01D_MN.rnx.zip",
                                     test_directory,)
    # nav data from previous day
    copy_data_file_to_test_directory("datasets/TLSE_2022001/BRDC00IGS_R_20213650000_01D_MN.rnx.gz",
                                     test_directory, )

    yield {"mixed_obs_file": test_mixed_obs_file,}
    shutil.rmtree(test_directory)


def test_prx_command_line_call_with_jsonseq_output(input_for_test):
    test_file = input_for_test
    prx_path = helpers.prx_root().joinpath("minimumLovablePrototype").joinpath("prx.py")
    command = f"python {prx_path} --observation_file_path {test_file}"
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxJsonTextSequenceFileExtension)
    )
    assert result.returncode == 0
    assert expected_prx_file.exists()


def test_prx_function_call_with_jsonseq_output(input_for_test):
    test_file = input_for_test
    prx.process(observation_file_path=test_file, output_format="jsonseq")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxJsonTextSequenceFileExtension)
    )
    assert expected_prx_file.exists()


def test_prx_function_call_with_csv_output(input_for_test):
    test_file = input_for_test
    prx.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()


def test_prx_function_call_for_obs_file_accross_two_days(input_for_test_with_first_epoch_at_midnight):
    test_file = input_for_test_with_first_epoch_at_midnight["mixed_obs_file"]
    prx.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()
