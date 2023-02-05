import os
from pathlib import Path
import shutil
import subprocess
import prx

import constants
import aux_files as aux


def test_directory():
    return Path(f"./tmp_test_directory_{__name__}").resolve()


def set_up_test():
    if test_directory().exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory())
    os.makedirs(test_directory())
    test_file = test_directory().joinpath("TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz")
    shutil.copy(prx.prx_root().joinpath(f"datasets/{test_file.name}"),
                test_file)
    assert test_file.exists()
    return test_file


def clean_up():
    shutil.rmtree(test_directory())


def test_find_local_ephemeris_file():
    test_file = set_up_test()
    aux_files = aux.get_on_it(test_file)
    assert type(aux_files) is dict
    shutil.rmtree(test_file.parent)
