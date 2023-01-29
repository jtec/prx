import os
from pathlib import Path
import shutil

import converters
import prx


def test_compressed_crx_to_rnx():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files have not been generated before and are still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz"
    shutil.copy(prx.prx_root().joinpath(f"datasets/{compressed_compact_rinex_file}"),
                test_directory.joinpath(compressed_compact_rinex_file))
    rinex_3_file = converters.anything_to_rinex_3(test_directory.joinpath(compressed_compact_rinex_file))
    expected_uncompacted_rinex_3_file = test_directory.joinpath(compressed_compact_rinex_file.replace('crx.gz', 'rnx'))
    assert rinex_3_file is not None
    assert rinex_3_file.exists()
    assert rinex_3_file == expected_uncompacted_rinex_3_file
    shutil.rmtree(test_directory)
