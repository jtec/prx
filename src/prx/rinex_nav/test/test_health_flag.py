import os
from pathlib import Path
import shutil

import pandas as pd
import pytest
import subprocess

from prx import util
from prx.rinex_nav import nav_file_discovery as aux
import georinex
from prx import converters
from prx.rinex_nav.evaluate import parse_rinex_nav_file, select_ephemerides


@pytest.fixture
def input_for_test():
    test_directory = (
        Path(__file__).parent.joinpath(f"./tmp_test_directory_{__name__}").resolve()
    )
    if test_directory.exists():
        # Start from empty directory, might avoid hiding some subtle bugs, e.g.
        # file decompression not working properly
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    test_files = {
        "prx_file": test_directory / "TLSE00FRA_R_20220010000_01H_30S_MO.csv",
        "rinex_nav_file": test_directory / "BRDC00IGS_R_20220010000_01D_MN.zip",
        "sp3_file": test_directory / "WUM0MGXULT_20220010000_01D_05M_ORB.SP3",
    }
    for key, test_file_path in test_files.items():
        shutil.copy(
            Path(__file__).parent.joinpath("datasets", test_file_path.name),
            test_file_path,
        )
        assert test_file_path.exists()
    yield test_files
    shutil.rmtree(test_directory)


def test_health_flag(input_for_test):
    """
    Tests the extraction and association of the health flag from a RINEX NAV file on a PRX DataFrame of epochs and satellites.
    """
    # read csv file
    df_prx = pd.read_csv(input_for_test["prx_file"], comment="#")

    # create query dataframe by extracting the right columns
    query = pd.DataFrame(
        {
            "sv": df_prx.constellation + df_prx.prn.astype(str).str.zfill(2),
            "signal": df_prx.rnx_obs_identifier,
            "query_time_isagpst": [
                pd.Timestamp(time) for time in df_prx.time_of_reception_in_receiver_time
            ],
        }
    )

    # read nav file
    rinex_nav_file = converters.compressed_to_uncompressed(
        input_for_test["rinex_nav_file"]
    )
    ephemerides = parse_rinex_nav_file(rinex_nav_file)

    # select the right ephemerides dataset for each query row
    query = select_ephemerides(ephemerides, query)

    # get health flag, according to constellation
    col_dict = {
        "G": "health",
        "E": "health",
        "C": "SatH1",
        "R": "health",  
        "J": "health",  
        "I": "Health", 
    }

    health_valid_ranges = {
        "G" : (0, 63),
        "E" : (0, 3),
        "C" : (0, 3),
        "R" : (0, 7),
        "J" : (0, 63),
        "I" : (0, 3),
    }
    # health_flag = [
    #     row._asdict()[col_dict[row.sv[0]]] for row in query.itertuples(index=False)
    # ]

    health_flag = []
    for row in query.itertuples(index=False):
        if row.sv[0] not in col_dict:
            pytest.fail(f"Constellation {row.sv[0]} missing")
        if not hasattr(row, col_dict[row.sv[0]]):
            pytest.fail(f"Parameter {col_dict[row.sv[0]]} missing")
        health_flag.append(getattr(row,col_dict[row.sv[0]] ))

    # merge health flag into prx dataframe
    df_prx["health_flag"] = health_flag

    # save prx dataframe as csv file
    df_prx.to_csv(  # verify parameters
        path_or_buf=(
            input_for_test["prx_file"].parent.joinpath(
                input_for_test["prx_file"].stem
                + "_new"
                + input_for_test["prx_file"].suffix
            )
        ),
        index=False,
        mode="a",
        float_format="%.6f",
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )

    # assert result
    # Verification : pas de NaN dans health_flah
    assert not df_prx["health_flag"].isna().any()

    # Vérification : valeurs dans plage de valeur possible
    assert all(
        health_valid_ranges[const][0] <= hf <= health_valid_ranges[const][1]
        for const, hf in zip(df_prx["constellation"], df_prx["health_flag"])
    )

    # Vérifier la valeur pour un échantillons de query

    print("done")
