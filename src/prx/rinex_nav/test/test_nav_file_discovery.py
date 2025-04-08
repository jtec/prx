import os
from pathlib import Path
import shutil

import pandas as pd
import pytest
import subprocess
import georinex
from prx.rinex_nav import nav_file_discovery as aux
from prx import helpers, converters


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
        helpers.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_obs_file.name}",
        test_obs_file,
    )
    shutil.copy(
        helpers.prx_repository_root()
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
        helpers.prx_repository_root() / "src/prx/rinex_nav/nav_file_discovery.py"
    )

    command = f"python {aux_file_script_path} --observation_file_path {test_file}"
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    assert result.returncode == 0


def test_wrong_name_for_local_nav(set_up_test):
    """
    Create a file in the same folder that has a non-conventional RNX3 NAV filename.
    Assert that prx will use a local or downloaded valid RNX NAV file.
    """
    # rename the nav file with a non-conventional name
    set_up_test["test_nav_file"].rename(
        Path(set_up_test["test_nav_file"].parent, "myfile.rnx")
    )
    aux_files = aux.discover_or_download_auxiliary_files(set_up_test["test_obs_file"])
    assert isinstance(aux_files, dict)
    assert "20230010000_01D_MN.rnx" in aux_files["broadcast_ephemerides"][0].name


def test_use_local_nav(set_up_test):
    """
    If the OBS file's folder contains a valid NAV folder, prx should choose the local NAV from the user folder rather
    than the one coming from its local database
    """
    local_file = converters.compressed_to_uncompressed(set_up_test["test_nav_file"])
    # rename the nav file with a different RNX3-compliant name: 'TLSE00IGS_R_20230010000_01D_MN.rnx'
    new_local_file = Path(local_file.parent, "TLSE" + local_file.name[4:])
    assert aux.is_rinex_3_mixed_mgex_broadcast_ephemerides_file(new_local_file)
    local_file.rename(new_local_file)
    aux_files = aux.discover_or_download_auxiliary_files(set_up_test["test_obs_file"])

    # get NAV file in local database
    local_database_file = aux.get_local_ephemerides(
        pd.Timestamp(year=int(local_file.name[12:16]), month=1, day=1)
        + pd.Timedelta(value=int(local_file.name[16:19]) - 1, unit="days")
    )
    if local_database_file is None:  # if BRDC NAV file is not in local database
        # download BRDC Nav file
        header = georinex.rinexheader(set_up_test["test_obs_file"])
        t_start = helpers.rinex_header_time_string_2_timestamp_ns(
            header["TIME OF FIRST OBS"]
        ) - pd.Timedelta(200, unit="milliseconds")
        t_end = helpers.rinex_header_time_string_2_timestamp_ns(
            header["TIME OF LAST OBS"]
        )
        aux.update_local_database(t_start, t_end)
    assert local_database_file.exists()

    # whether a NAV file exists in the local database, the local user NAV file should be chosen if it is compliant with
    # RNX3 naming convention
    assert isinstance(aux_files, dict)
    assert new_local_file.name == aux_files["broadcast_ephemerides"][0].name
