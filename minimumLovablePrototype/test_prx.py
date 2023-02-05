import os
from pathlib import Path
import shutil
import subprocess

import prx
import constants


def test_prx_command_line_call_with_jsonseq_output():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz"
    shutil.copy(prx.prx_root().joinpath(f"datasets/{compressed_compact_rinex_file}"),
                test_directory.joinpath(compressed_compact_rinex_file))
    prx_path = prx.prx_root().joinpath("minimumLovablePrototype").joinpath("prx.py")
    command = f"python {prx_path} --observation_file_path {test_directory.joinpath(compressed_compact_rinex_file)}"
    result = subprocess.run(command, capture_output=True, shell=True, cwd=test_directory)
    expected_prx_file = test_directory.joinpath(
        compressed_compact_rinex_file.replace('crx.gz', constants.cPrxJsonTextSequenceFileExtension))
    assert result.returncode == 0
    assert expected_prx_file.exists()
    shutil.rmtree(test_directory)


def test_prx_function_call_with_jsonseq_output():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    rinex_observation_file = "TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz"
    shutil.copy(prx.prx_root().joinpath(f"datasets/{rinex_observation_file}"),
                test_directory.joinpath(rinex_observation_file))
    assert test_directory.joinpath(rinex_observation_file).exists()
    prx.process(observation_file_path=test_directory.joinpath(rinex_observation_file), output_format="jsonseq")
    expected_prx_file = test_directory.joinpath(
        rinex_observation_file.replace('rnx', constants.cPrxJsonTextSequenceFileExtension))
    assert expected_prx_file.exists()
    shutil.rmtree(test_directory)
