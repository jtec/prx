import os
from pathlib import Path
import shutil
import pytest
import subprocess

from prx import util
from prx.rinex_nav import nav_file_discovery as aux


@pytest.fixture
def set_up_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_obs_file = test_directory.joinpath("TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz")
    test_nav_file = test_directory.joinpath("BRDC00IGS_R_20230010000_01D_MN.rnx.zip")

    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_obs_file.name}",
        test_obs_file,
    )
    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_nav_file.name}",
        test_nav_file,
    )

    assert test_obs_file.exists()
    assert test_nav_file.exists()

    yield {"test_obs_file": test_obs_file, "test_nav_file": test_nav_file}
    shutil.rmtree(test_directory)


def test_find_local_ephemeris_file(set_up_test):
    aux_files = aux.discover_or_download_auxiliary_files(set_up_test["test_obs_file"])
    assert isinstance(aux_files, dict)


def test_download_remote_ephemeris_files(set_up_test):
    os.remove(set_up_test["test_nav_file"])
    aux_files = aux.discover_or_download_auxiliary_files(set_up_test["test_obs_file"])
    assert isinstance(aux_files, dict)


def test_command_line_call(set_up_test):
    test_file = set_up_test["test_obs_file"]
    aux_file_script_path = (
        util.prx_repository_root() / "src/prx/rinex_nav/nav_file_discovery.py"
    )

    command = f"python {aux_file_script_path} --observation_file_path {test_file}"
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    assert result.returncode == 0
