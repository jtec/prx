import os
from pathlib import Path
import shutil
import georinex
import pytest

import pandas as pd
from unittest.mock import patch 
import prx
from prx import util
from prx.precise_corrections.sp3 import sp3_file_discovery as sp3


@pytest.fixture
def set_up_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected files has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_obs_file = test_directory.joinpath("TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz")
    # local_database = sp3.sp3_file_folder(pd.Timestamp('2023-01-01'), Path('src/prx/precise_corrections/sp3/test/datasets'))

    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{test_obs_file.name}",
        test_obs_file,
    )
    assert test_obs_file.exists()
    test_date = pd.Timestamp("2023-01-01")
    sp3_subfolder = sp3.sp3_file_folder(test_date, test_directory)
    test_sp3_file = sp3_subfolder / "WUM0MGXFIN_20230010000_01D_05M_ORB.SP3.gz"
    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/precise_corrections/sp3/test/datasets/2023/001/{test_sp3_file.name}",
        test_sp3_file,
    )
    assert test_sp3_file.exists()

    test_clk_file = sp3_subfolder / 'WUM0MGXFIN_20230010000_01D_30S_CLK.CLK.gz'
    shutil.copy(
        util.prx_repository_root()
        / f"src/prx/precise_corrections/sp3/test/datasets/2023/001/{test_clk_file.name}",
        test_clk_file,
    )
    assert test_clk_file.exists()

    yield {"test_obs_file": test_obs_file, "test_sp3_file" : test_sp3_file, 'test_clk_file' : test_clk_file}
    shutil.rmtree(test_directory)

def test_get_index_of_priority_from_filename(set_up_test):
    obs_file = set_up_test["test_obs_file"]
    sp3_filename = set_up_test["test_sp3_file"].name
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    index_in_priority = sp3.get_index_of_priority_from_filename(sp3_filename, t_start)

    assert index_in_priority == 6

def test_get_sp3_file(set_up_test):
    """
    Assert that prx will download the first SP3 file in the priority list if not present.
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    t_end = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])

    db_folder = set_up_test["test_obs_file"].parent

    sp3.get_sp3_file(t_start, t_end, db_folder)

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(t_start)
    expected_priority = sp3.priority
    
    sp3_filename, clk_filename = sp3.build_sp3_filename(gps_week, dow, t_start, expected_priority[0], expected_priority)
    sp3_filename = str(Path(sp3_filename).with_suffix(""))
    clk_filename = str(Path(clk_filename).with_suffix(""))

    downloaded_sp3 = sp3.get_local_sp3(t_start, sp3_filename, db_folder)
    downloaded_clk = sp3.get_local_sp3(t_start, clk_filename, db_folder)
    assert downloaded_sp3 is not None
    assert downloaded_clk is not None 
    assert downloaded_sp3.exists()
    assert downloaded_clk.exists()

def test_priority_local_file_found_midway(set_up_test):
    '''
    This test verifies that `get_sp3_file` respects the SP3/CLK priority list when attempting to
    retrieve orbit and clock files, stopping at the first valid local pair.

    The test simulates the following scenario across increasing priority lengths:

    1. When priority = [('COD', 'FIN')]:
        - Local file at priority 0 is missing → triggers FTP download.
        - Download succeeds → file becomes available locally.
        - `get_sp3_file` finds and returns the local file.

    2. When priority = [('COD', 'FIN'), ('GRG', 'FIN')]:
        - Local 0 is missing → download fails.
        - Local 1 is missing → triggers download.
        - Download of 1 succeeds → file becomes available locally.
        - `get_sp3_file` returns the local file at priority 1.

    This is done until the local files are found 

    This test ensures:
    - The function checks local availability before attempting download.
    - Each priority is tried in order until a valid file is found.
    - The function stops as soon as a valid local file is available (even if previously downloaded).
    - The logic is robust regardless of which priority index the file becomes available at.
    '''

    t_start = pd.Timestamp("2023-05-14 12:00:00")

    db_folder = set_up_test["test_obs_file"].parent
    expected_priority = sp3.priority

    index_found_file_priority = len(expected_priority)

    for i in range(index_found_file_priority):
        for j in range(i+1):
            if i != j :
                with patch("prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp", return_value = None), \
                     patch("prx.precise_corrections.sp3.sp3_file_discovery.priority", expected_priority[:i+1]):
                        sp3_file, clk_file = sp3.get_sp3_file(t_start, t_start, db_folder) 
                assert (sp3_file, clk_file) == (None, None)

            else : 
                with patch("prx.precise_corrections.sp3.sp3_file_discovery.priority", [expected_priority[i]]):
                    sp3_file, clk_file = sp3.get_sp3_file(t_start, t_start, db_folder)
                    if sp3_file is None :
                        print(f"Could not download {sp3_file}")
                    if clk_file is None : 
                        print(f"Could not download {clk_file}")
                    elif sp3_file.exists() :
                        sp3_file.unlink()
                    elif clk_file.exists():
                        clk_file.unlink() 

def test_priority_iteration_until_matching_sp3_clk_pair_found(set_up_test):
    '''
    This test ensures that `get_sp3_file` correctly iterates over the list of SP3/CLK priorities 
    when searching for a matching orbit/clock file pair.

    The scenario being tested simulates the following:

    - A local SP3 file is available at priority index 4.
    - A local CLK file is available at priority index 5.

    The test walks through increasing slices of the priority list and validates the following:

    1. For all priority levels (0, 0) up to (4, 4):
        - Matching pairs are downloaded (if the files exist).
        - `get_sp3_file` returns the pair.

    2. At priority index 5:
        - The function finds the valid (SP3, CLK) pair locally.
        - `get_sp3_file` returns the correct local files and terminates.

    '''
    t_start = pd.Timestamp('2023-06-17 12:00:00')
    
    db_folder = set_up_test["test_obs_file"].parent

    expected_priority =sp3.priority
    
    index_priority_sp3 = 4
    index_priority_clk = 5
    index_iteration = max(index_priority_sp3, index_priority_clk)

    for i in range(index_iteration+1):
        for j in range(i+1):
            if i != j and i != index_iteration:
                with patch("prx.precise_corrections.sp3.sp3_file_discovery.try_downloading_sp3_ftp", return_value = None), \
                     patch("prx.precise_corrections.sp3.sp3_file_discovery.priority", expected_priority[:i+1]):
                        sp3_file, clk_file = sp3.get_sp3_file(t_start, t_start, db_folder) 
                assert (sp3_file, clk_file) == (None, None)
            elif i == index_iteration :
                with patch("prx.precise_corrections.sp3.sp3_file_discovery.priority", [expected_priority[i]]):
                    sp3_file, clk_file = sp3.get_sp3_file(t_start, t_start, db_folder) 
                assert sp3_file is not None
                assert clk_file is not None
            else : 
                with patch("prx.precise_corrections.sp3.sp3_file_discovery.priority", [expected_priority[i]]):
                    sp3_file, clk_file = sp3.get_sp3_file(t_start, t_start, db_folder)
                    if sp3_file is None :
                        print(f"Could not download {sp3_file}")
                    if clk_file is None : 
                        print(f"Could not download {clk_file}")
                    elif sp3_file.exists() :
                        sp3_file.unlink()
                    elif clk_file.exists():
                        clk_file.unlink()
                assert sp3.get_index_of_priority_from_filename(str(Path(sp3_file).with_suffix(""))) == sp3.get_index_of_priority_from_filename(str(Path(clk_file).with_suffix("")))

def test_download_all(set_up_test):
    """
    At least one file was successfully downloaded 
    """
    obs_file = set_up_test["test_obs_file"]
    header = georinex.rinexheader(obs_file)
    t_start = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF FIRST OBS"]) - pd.Timedelta(200, unit="milliseconds")
    t_end = util.rinex_header_time_string_2_timestamp_ns(header["TIME OF LAST OBS"])

    db_folder = set_up_test["test_obs_file"].parent

    gps_week, dow = sp3.timestamp_to_gps_week_and_dow(t_start)
    
    priority = sp3.priority

    downloaded_sp3_files = []
    downloaded_clk_files = []
    for p in priority:
        sp3_filename, clk_filename = sp3.build_sp3_filename(gps_week, dow, t_start, p, priority)
        downloaded_sp3 = sp3.try_downloading_sp3_ftp(gps_week, t_start, db_folder, sp3_filename)
        downloaded_clk = sp3.try_downloading_sp3_ftp(gps_week, t_start, db_folder, clk_filename)
        
        if downloaded_sp3 is not None :
            downloaded_sp3_files.append(downloaded_sp3)
        if downloaded_clk is not None :
            downloaded_clk_files.append(downloaded_clk)
        
    assert downloaded_sp3_files
    assert downloaded_clk_files




