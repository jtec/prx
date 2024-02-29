
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
    # Elevation sanity check
    assert (
        df[(df.prn == 14) & (df.constellation == "C")].sat_elevation_deg - 34.86
    ).abs().max() < 0.3







def prx_csv_to_pandas(filepath: str):
    """
    # Read a PRX_CSV file and convert it to pandas DataFrame
    """
    data_prx = pd.read_csv(
        filepath,
        comment="#",

        parse_dates=["time_of_reception_in_receiver_time"],
    )
    return data_prx

def test_csv_format():
    # Path to the generated prx csv file
    csv_file = (
        helpers.prx_repository_root()
        .joinpath("tools/validation_data/TLSE00FRA_R_20230010100_10S_01S_MO.csv")
    )
    # Read the generated CSV file and convert it to pandas DataFrame
    data_prx = prx_csv_to_pandas(csv_file)

    # List of expected field names after renaming
    fields_expected = ['sat_clock_offset_m', 'sat_clock_drift_mps', 'sat_pos_x_m','sat_vel_x_mps',
    'sat_instrumental_delay_m','iono_delay_m','sat_elevation_deg','sat_azim_deg','rnx_obs_identifier','C_obs_m']

    # Check if the renamed fields are present in the DataFrame columns
    for field in fields_expected:
        assert field in data_prx.columns

    # Check values of the first observation for a few fields (not involving prx computations)
    assert data_prx.iloc[0].time_of_reception_in_receiver_time == pd.Timestamp("2023-01-01 01:00:00")
    assert data_prx.iloc[0].constellation == "C"
    assert data_prx.iloc[0].prn == 5
    assert data_prx.iloc[0].observation_code == "2I"
    assert data_prx.iloc[0].code_observation_m == 39902331.27300
    assert data_prx.iloc[0].doppler_observation_hz == -19.17200
    assert data_prx.iloc[0].carrier_observation_m == 39902329.09610697
    assert data_prx.iloc[0].cn0_dbhz == 35.60000
