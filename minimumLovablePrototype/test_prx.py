import os
import time
import random
from pathlib import Path
import shutil
import constants

import prx


def test_prx_processing_with_jsonseq_output():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and are still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    rinex_observation_file = "TLSE00FRA_R_20230010000_03S_01S_MO.rnx"
    shutil.copy(prx.prx_root().joinpath(f"datasets/{rinex_observation_file}"),
                test_directory.joinpath(rinex_observation_file))
    assert test_directory.joinpath(rinex_observation_file).exists()
    prx.process(test_directory.joinpath(rinex_observation_file), "jsonseq")
    expected_prx_file = test_directory.joinpath(rinex_observation_file.replace('rnx', constants.cPrxJsonTextSequenceFileExtension))
    assert expected_prx_file.exists()
    shutil.rmtree(test_directory)
