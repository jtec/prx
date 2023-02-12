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
    test_obs_file = test_directory.joinpath("TLSE00FRA_R_20230010000_10S_01S_MO.crx.gz")
    test_nav_file = test_directory.joinpath("BRDC00IGS_R_20230010000_01D_MN.rnx.gz")

    shutil.copy(
        prx.prx_root().joinpath(f"datasets/TLSE_2023001/{test_obs_file.name}"),
        test_obs_file,
    )
    shutil.copy(
        prx.prx_root().joinpath(f"datasets/TLSE_2023001/{test_nav_file.name}"),
        test_nav_file,
    )

    assert test_obs_file.exists()
    assert test_nav_file.exists()

    yield {"test_obs_file": test_obs_file, "test_nav_file": test_nav_file}
    shutil.rmtree(test_directory)


def test_find_local_ephemeris_file(set_up_test):
    aux_files = aux.get_on_it(set_up_test["test_obs_file"])
    assert type(aux_files) is dict


def test_download_remote_ephemeris_files(set_up_test):
    os.remove(set_up_test["test_nav_file"])
    aux_files = aux.get_on_it(set_up_test["test_obs_file"])
    assert type(aux_files) is dict
