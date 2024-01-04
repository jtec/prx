import os
import time
import random
import src.prx.helpers as helpers
from pathlib import Path
import pandas as pd
import numpy as np

# import csv_example

# def test_csv_file_generated():
#     random.seed(time.time())
#     # Leverage the file name argument to generate a file with a unique name and delete it right afterwards:
#     test_file_path = (
#         Path("__file__")
#         .resolve()
#         .parent.joinpath(Path(f"tmp_test_file_{random.randint(int(0), int(1e12))}.csv"))
#     )
#     # Be extra careful and check that the file does not already exist before we try to generate it:
#     assert not test_file_path.exists()
#     csv_example.generate_example_file(test_file_path)
#     assert test_file_path.exists()
#     assert os.path.getsize(test_file_path) > 0
#     os.remove(test_file_path)


def converter_string_to_array(string):
    # convert a string "[a,b,c]" to np.array([a,b,c])
    return np.fromstring(string[1:-1], sep=" ")

def prx_csv_to_pandas(filepath: str):
    """
    Read a PRX_CSV file and convert it to pandas DataFrame
    """
    data_prx = pd.read_csv(
        filepath,
        comment="#",
        converters={
            "satellite_position_m": converter_string_to_array,
            "satellite_velocity_mps": converter_string_to_array,
            "approximate_antenna_position_m": converter_string_to_array,
        },
        parse_dates=["time_of_reception_in_receiver_time"],
    )
    return data_prx


def test_csv_format():
    # TODO change path to a generated prx csv file
    csv_file = (
        helpers.prx_repository_root()
        .joinpath("tools/validation_data/TLSE00FRA_R_20230010100_10S_01S_MO.csv")
    )

    # read csv_file and convert it to pandas.DataFrame
    data_prx = prx_csv_to_pandas(csv_file)

    # check prx columns
    fields_csv = data_prx.columns
    fields_expected = ['time_of_reception_in_receiver_time', 'constellation', 'prn',
       'observation_code', 'code_observation_m', 'doppler_observation_hz',
       'carrier_observation_m', 'lli', 'cn0_dbhz', 'satellite_position_m',
       'satellite_velocity_mps', 'satellite_clock_bias_m',
       'satellite_clock_bias_drift_mps', 'sagnac_effect_m',
       'relativistic_clock_effect_m', 'group_delay_m', 'iono_delay_m',
       'tropo_delay_m', 'approximate_antenna_position_m']
    for field in fields_expected:
        assert(field in fields_csv)

    # check values of the first observation for a few fields (not involving prx computations)
    assert( data_prx.iloc[0].time_of_reception_in_receiver_time == pd.Timestamp("2023-01-01 01:00:00") )
    assert( data_prx.iloc[0].constellation == "C" )
    assert( data_prx.iloc[0].prn == 5 )
    assert( data_prx.iloc[0].observation_code == "2I" )
    assert( data_prx.iloc[0].code_observation_m == 39902331.27300 )
    assert( data_prx.iloc[0].doppler_observation_hz == -19.17200 )
    assert( data_prx.iloc[0].carrier_observation_m == 39902329.09610697 )
    assert( data_prx.iloc[0].cn0_dbhz == 35.60000 )
