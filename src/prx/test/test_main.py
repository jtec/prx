import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd
import pytest

from prx import helpers
from prx import constants
from prx import main


# This function sets up a temporary directory, copies a rinex observations file into that directory
# and returns its path. The @pytest.fixture annotation allows us to pass the function as an input
# to test functions. When running a test function, pytest will then first run this function, pass
# whatever is passed to `yield` to the test function, and run the code after `yield` after the test,
# even  if the test crashes.
@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    test_file = test_directory.joinpath(compressed_compact_rinex_file)
    shutil.copy(
        Path(__file__).parent
        / f"datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_file,
    )
    assert test_file.exists()
    # Also provide ephemerides so the test does not have to download them:
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{ephemerides_file}",
        test_file.parent.joinpath(ephemerides_file),
    )
    assert test_file.parent.joinpath(ephemerides_file).exists()

    yield test_file
    # shutil.rmtree(test_file.parent)


def test_prx_command_line_call_with_csv_output(input_for_test):
    test_file = input_for_test
    prx_path = helpers.prx_repository_root() / "src/prx/main.py"
    command = (
        f"python {prx_path} --observation_file_path {test_file} --output_format csv"
    )
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert result.returncode == 0
    assert expected_prx_file.exists()


def test_prx_function_call_with_csv_output(input_for_test):
    test_file = input_for_test
    main.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()
    df = pd.read_csv(expected_prx_file, comment="#")
    assert not df.empty
    assert helpers.is_sorted(df.time_of_reception_in_receiver_time)
    # Elevation sanity check
    assert (
        df[(df.prn == 14) & (df.constellation == "C")].elevation_deg - 34.86
    ).abs().max() < 0.3


def test_spp_lsq(input_for_test):
    test_file = input_for_test
    main.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()
    df = pd.read_csv(expected_prx_file, comment="#")
    assert not df.empty
    df.group_delay_m = df.group_delay_m.fillna(0)
    df = df[df.C_obs.notna() & df.x_m.notna()]
    df_first_epoch = df[df.time_of_reception_in_receiver_time == df.time_of_reception_in_receiver_time.min()]
    df_first_epoch['C_obs_corrected'] = (df_first_epoch.C_obs
                                         - df_first_epoch.clock_m
                                         - df_first_epoch.relativistic_clock_effect_m
                                         - df_first_epoch.sagnac_effect_m
                                         - df_first_epoch.code_iono_delay_klobuchar_m
                                         - df_first_epoch.tropo_delay_m
                                         - df_first_epoch.group_delay_m)
    pass