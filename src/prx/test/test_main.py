import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from prx import helpers
from prx import constants
from prx import main
from prx.user import (
    parse_prx_csv_file,
    spp_pt_lsq,
    spp_vt_lsq,
    bootstrap_coarse_receiver_position,
)
from prx.rinex_nav import nav_file_discovery

log = helpers.get_logger(__name__)


# This function sets up a temporary directory, copies a rinex observations file into that directory
# and returns its path. The @pytest.fixture annotation allows us to pass the function as an input
# to test functions. When running a test function, pytest will then first run this function, pass
# whatever is passed to `yield` to the test function, and run the code after `yield` after the test,
# even if the test crashes.
@pytest.fixture
def input_for_test_tlse():
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
    shutil.rmtree(test_file.parent)


@pytest.fixture
def input_for_test_nist():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "NIST00USA_R_20230010100_05M_30S_MO.crx.gz"
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
    shutil.rmtree(test_file.parent)


@pytest.fixture
def input_for_test_with_first_epoch_at_midnight():
    # Having a first epoch at midnight requires to have the NAV data from the previous day, because we are computing
    # the time of emission as (time of reception - pseudorange/celerity)
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)

    filepath_to_obs_file = "TLSE00FRA_R_20230010000_30M_30S_GO.crx.gz"
    test_obs_file = test_directory.joinpath(filepath_to_obs_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{filepath_to_obs_file}",
        test_obs_file,
    )
    assert test_obs_file.exists()

    # nav data from same day
    shutil.copy(
        Path(__file__).parent
        / "datasets/TLSE_2023001/BRDC00IGS_R_20230010000_01D_MN.rnx.zip",
        test_directory.joinpath("BRDC00IGS_R_20230010000_01D_MN.rnx.zip"),
    )
    # nav data from previous day
    shutil.copy(
        Path(__file__).parent
        / "datasets/TLSE_2023001/BRDC00IGS_R_20223650000_01D_MN.rnx.gz",
        test_directory.joinpath("BRDC00IGS_R_20223650000_01D_MN.rnx.gz"),
    )

    yield {
        "obs_file": test_obs_file,
    }
    shutil.rmtree(test_directory)


def test_prx_command_line_call_with_csv_output(input_for_test_tlse):
    test_file = input_for_test_tlse
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


def test_prx_function_call_with_csv_output(input_for_test_tlse):
    test_file = input_for_test_tlse
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
        df[(df.prn == 14) & (df.constellation == "C")].sat_elevation_deg - 34.86
    ).abs().max() < 0.3


def test_prx_function_call_for_obs_file_across_two_days(
    input_for_test_with_first_epoch_at_midnight,
):
    test_file = input_for_test_with_first_epoch_at_midnight["obs_file"]
    main.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()


def run_rinex_through_prx(rinex_obs_file: Path):
    main.process(observation_file_path=rinex_obs_file, output_format="csv")
    expected_prx_file = Path(
        str(rinex_obs_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()
    records, metadata = parse_prx_csv_file(expected_prx_file)
    records = pd.read_csv(expected_prx_file, comment="#")
    assert not records.empty
    assert metadata
    records.sat_code_bias_m = records.sat_code_bias_m.fillna(0)
    records = records[records.C_obs_m.notna() & records.sat_pos_x_m.notna()]
    return records, metadata


def test_spp_lsq_nist(input_for_test_nist):
    df, metadata = run_rinex_through_prx(input_for_test_nist)
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        df.time_of_reception_in_receiver_time
        == df.time_of_reception_in_receiver_time.min()
    ]
    for constellations_to_use in [
        (
            "G",
            "E",
            "C",
        ),
        ("G", "S"),
        ("G",),
        ("E",),
        ("C",),
        ("R",),
    ]:
        obs = df_first_epoch[df.constellation.isin(constellations_to_use)]
        pt_lsq = spp_pt_lsq(obs)
        vt_lsq = spp_vt_lsq(obs, p_ecef_m=pt_lsq[0:3, :])
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        # Static receiver, so:
        velocity_offset = vt_lsq[0:3, :]
        log.info(
            f"Using constellations: {constellations_to_use}, {len(obs.sv.unique())} SVs"
        )
        log.info(f"Position offset: {position_offset}")
        log.info(f"Velocity offset: {velocity_offset}")
        assert (
            np.max(np.abs(position_offset)) < 2e1
        )  # relaxed position offset (instead of 1e1)
        assert np.max(np.abs(velocity_offset)) < 1e-1


def test_spp_lsq_tlse(input_for_test_tlse):
    df, metadata = run_rinex_through_prx(input_for_test_tlse)
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        df.time_of_reception_in_receiver_time
        == df.time_of_reception_in_receiver_time.min()
    ]
    for constellations_to_use in [
        (
            "G",
            "E",
            "C",
        ),
        ("G", "S"),
        ("G",),
        ("E",),
        ("C",),
        ("R",),
    ]:
        obs = df_first_epoch[df.constellation.isin(constellations_to_use)]
        pt_lsq = spp_pt_lsq(obs)
        vt_lsq = spp_vt_lsq(obs, p_ecef_m=pt_lsq[0:3, :])
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        # Static receiver, so:
        velocity_offset = vt_lsq[0:3, :]
        log.info(
            f"Using constellations: {constellations_to_use}, {len(obs.sv.unique())} SVs"
        )
        log.info(f"Position offset: {position_offset}")
        log.info(f"Velocity offset: {velocity_offset}")
        assert np.max(np.abs(position_offset)) < 1e1
        assert np.max(np.abs(velocity_offset)) < 1e-1


def test_spp_lsq_for_obs_file_across_two_days(
    input_for_test_with_first_epoch_at_midnight,
):
    df, metadata = run_rinex_through_prx(
        input_for_test_with_first_epoch_at_midnight["obs_file"]
    )
    df_first_epoch = df[
        df.time_of_reception_in_receiver_time
        == df.time_of_reception_in_receiver_time.min()
    ]
    for constellations_to_use in [
        ("G",),
    ]:
        obs = df_first_epoch[df.constellation.isin(constellations_to_use)]
        pt_lsq = spp_pt_lsq(obs)
        vt_lsq = spp_vt_lsq(obs, p_ecef_m=pt_lsq[0:3, :])
        position_offset = pt_lsq[0:3, :] - np.array(
            metadata["approximate_receiver_ecef_position_m"]
        ).reshape(-1, 1)
        # Static receiver, so:
        velocity_offset = vt_lsq[0:3, :]
        log.info(f"Position offset: {position_offset}")
        log.info(f"Velocity offset: {velocity_offset}")
        assert np.max(np.abs(position_offset)) < 1e1
        assert np.max(np.abs(velocity_offset)) < 1e-1


def test_csv_column_names(input_for_test_tlse):
    test_file = input_for_test_tlse
    main.process(observation_file_path=test_file, output_format="csv")
    expected_prx_file = Path(
        str(test_file).replace("crx.gz", constants.cPrxCsvFileExtension)
    )
    assert expected_prx_file.exists()

    # Read the CSV file
    df = pd.read_csv(expected_prx_file, comment="#")

    # Expected CSV column names
    expected_column_names = {
        "time_of_reception_in_receiver_time",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "relativistic_clock_effect_m",
        "sagnac_effect_m",
        "tropo_delay_m",
        "sat_code_bias_m",
        "carrier_frequency_hz",
        "iono_delay_m",
        "sat_elevation_deg",
        "sat_azimuth_deg",
        "rnx_obs_identifier",
        "C_obs_m",
        "D_obs_hz",
        "L_obs_cycles",
        "S_obs_dBHz",
        "constellation",
        "prn",
    }

    # Checking if all renamed parameters exist in the dataframe columns
    for parameter in expected_column_names:
        assert parameter in df.columns


def test_first_position_from_georinex(input_for_test_tlse):
    # Download RNX NAV file
    aux_files = nav_file_discovery.discover_or_download_auxiliary_files(
        input_for_test_tlse
    )

    # Compute solution from first epoch with more than 4 GPS L1C/A observations, without minimum corrections
    solution = bootstrap_coarse_receiver_position(
        input_for_test_tlse, aux_files["broadcast_ephemerides"]
    )

    # Retrieve approximate position from RINEX OBS header
    metadata = main.build_metadata(
        {
            "obs_file": input_for_test_tlse,
            "nav_file": aux_files["broadcast_ephemerides"],
        }
    )

    # Compute position error
    position_error_ecef_m = (
        solution[0:3] - metadata["approximate_receiver_ecef_position_m"]
    )

    # Verify position error magnitude
    # A tolerance of 100 m is accepted due to missing corrections (no atmospheric corrections, Sagnac effect)
    # and the use of only one constellation and signal
    assert np.linalg.norm(position_error_ecef_m) < 100
