import os
from pathlib import Path
import shutil
import pytest
import prx
import aux_files as aux


@pytest.fixture
def set_up_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_file = test_directory.joinpath("TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz")
    shutil.copy(prx.prx_root().joinpath(f"datasets/{test_file.name}"), test_file)
    assert test_file.exists()
    yield {"test_file": test_file}
    shutil.rmtree(test_directory)


def test_find_local_ephemeris_file(set_up_test):
    aux_files = aux.get_on_it(set_up_test["test_file"])
    assert type(aux_files) is dict
