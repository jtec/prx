import os
import time
import random
from pathlib import Path
import shutil

import prx


def test_prx_processing():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and are still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz"
    shutil.copy(prx.prx_root().joinpath(f"datasets/{compressed_compact_rinex_file}"),
                test_directory.joinpath(compressed_compact_rinex_file))
    prx.process(test_directory.joinpath(compressed_compact_rinex_file))
    expected_prx_file = test_directory.joinpath(compressed_compact_rinex_file.replace('crx.gz', 'json'))
    assert expected_prx_file.exists()
    shutil.rmtree(test_directory)
