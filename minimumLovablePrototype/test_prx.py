import os
from pathlib import Path
import shutil
import subprocess

import pytest

import prx
import constants


def test_directory():
    return Path(f"./tmp_test_directory_{__name__}").resolve()


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
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz"
    test_file = test_directory.joinpath(compressed_compact_rinex_file)
    shutil.copy(
        prx.prx_root().joinpath(
            f"datasets/TLSE_2023001/{compressed_compact_rinex_file}"
        ),
        test_file,
    )
    assert test_file.exists()
    yield test_file
    shutil.rmtree(test_file.parent)


def test_prx_command_line_call_with_jsonseq_output(input_for_test):
    test_file = input_for_test
    prx_path = prx.prx_root().joinpath("minimumLovablePrototype").joinpath("prx.py")
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
