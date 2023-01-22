import os
import time
import random
from pathlib import Path
import shutil

import converters
import prx


def test_compressed_crx_to_rnx():
    test_directory = Path(f"./tmp_test_directory/{__name__}").resolve()
    os.makedirs(test_directory, exist_ok=True)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz"
    shutil.copy(prx.prx_root().joinpath(f"datasets/{compressed_compact_rinex_file}"),
                test_directory.joinpath(compressed_compact_rinex_file))
    converters.anything_to_rinex_3(test_directory.joinpath(compressed_compact_rinex_file))