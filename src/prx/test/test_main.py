import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from prx import util
from prx import main
from prx.user import (
    parse_prx_csv_file_metadata,
    parse_prx_csv_file,
    spp_pt_lsq,
    spp_vt_lsq,
    bootstrap_coarse_receiver_position,
)
from prx.rinex_nav import nav_file_discovery

log = util.get_logger(__name__)


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
    datasets_directory = Path(__file__).parent / "datasets"
    # Also provide ephemerides on disk so the test does not have to download them:
    compressed_compact_rinex_file = (
        datasets_directory
        / "TLSE_2023001"
        / "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    )
    ephemerides_file = (
        datasets_directory / "TLSE_2023001/BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
    )
    for file in [compressed_compact_rinex_file, ephemerides_file]:
        shutil.copy(
            file,
            test_directory / file.name,
        )

    yield test_directory / compressed_compact_rinex_file.name
    shutil.rmtree(test_directory)


@pytest.fixture
def input_for_test_tlse_2024():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    datasets_directory = Path(__file__).parent / "datasets"
    # Also provide ephemerides on disk so the test does not have to download them:
    compressed_compact_rinex_file = (
        datasets_directory
        / "TLSE00FRA_R_2024001"
        / "TLSE00FRA_R_20240011800_01H_30S_MO.crx.gz"
    )
    ephemerides_file = (
        datasets_directory / "TLSE00FRA_R_2024001/BRDC00IGS_R_20240010000_01D_MN.rnx.gz"
    )
    for file in [compressed_compact_rinex_file, ephemerides_file]:
        shutil.copy(
            file,
            test_directory / file.name,
        )

    yield test_directory / compressed_compact_rinex_file.name
    shutil.rmtree(test_directory)


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


def test_prx_command_line_call(input_for_test_tlse):
    test_file = input_for_test_tlse
    prx_path = util.prx_repository_root() / "src/prx/main.py"
    command = f"python {prx_path} --observation_file_path {test_file}"
    result = subprocess.run(
        command, capture_output=True, shell=True, cwd=str(test_file.parent)
    )
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    assert result.returncode == 0
    assert expected_prx_file.exists()


def test_prx_function_call(input_for_test_tlse):
    test_file = input_for_test_tlse
    main.process(observation_file_path=test_file, prx_level=2)
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()
    df = pd.read_csv(expected_prx_file, comment="#")
    assert not df.empty
    assert util.is_sorted(df.time_of_reception_in_receiver_time)
    # Elevation sanity check
    assert (
        df[(df.prn == 14) & (df.constellation == "C")].sat_elevation_deg - 34.86
    ).abs().max() < 0.3
    # In such as short period we expect at most one transition from one ephemeris to the next one
    for signal, group_df in df.groupby(["constellation", "prn", "rnx_obs_identifier"]):
        n_ephemeris_changes = sum(
            group_df.ephemeris_hash.astype(float).diff().dropna().abs() > 0
        )
        assert n_ephemeris_changes <= 1


def test_prx_lli_parsing(input_for_test_tlse):
    test_file = input_for_test_tlse
    main.process(observation_file_path=test_file)
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    df = pd.read_csv(expected_prx_file, comment="#")
    # LLI check - CS expected (LLI = 1)
    assert (
        df.loc[
            (df.time_of_reception_in_receiver_time == "2023-01-01 01:00:01.000000")
            & (df.rnx_obs_identifier == "7D")
            & (df.constellation == "C")
            & (df.prn == 30),
            "LLI",
        ]
        == 1
    ).all()
    assert (
        df.loc[
            (df.time_of_reception_in_receiver_time == "2023-01-01 01:00:07.000000")
            & (df.rnx_obs_identifier.isin(["2W", "1C"]))
            & (df.constellation == "G")
            & (df.prn == 17),
            "LLI",
        ]
        == 1
    ).all()
    # LLI check - no CS (LLI = 0)
    assert (
        df.loc[
            (df.time_of_reception_in_receiver_time == "2023-01-01 01:00:07.000000")
            & (df.rnx_obs_identifier == "2I")
            & (df.constellation == "C")
            & (df.prn == 5),
            "LLI",
        ]
        == 0
    ).all()
    # LLI check - no phase tracking (L_obs_cycles = NaN and LLI = NaN)
    assert df.loc[df.L_obs_cycles.isnull(), "LLI"].isnull().all()


def test_prx_function_call_for_obs_file_across_two_days(
    input_for_test_with_first_epoch_at_midnight,
):
    test_file = input_for_test_with_first_epoch_at_midnight["obs_file"]
    main.process(observation_file_path=test_file)
    expected_prx_file = Path(str(test_file).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()


def run_rinex_through_prx(rinex_obs_file: Path, prx_level: int = 2):
    main.process(observation_file_path=rinex_obs_file, prx_level=prx_level)
    expected_prx_file = Path(str(rinex_obs_file).replace("crx.gz", "csv"))
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
    assert (df.iono_delay_m > 0).all()
    assert (df.tropo_delay_m > 0).all()
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
        (
            df.time_of_reception_in_receiver_time
            == df.time_of_reception_in_receiver_time.min()
        )
        & (df.sat_elevation_deg > 10)
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


def test_spp_lsq_tlse_2024(input_for_test_tlse_2024):
    """
    Calculates the position offset twice.
    Once including a faulty satellite (G27) and once only with satellites with a health_flag of 0.

    Allows to assess the faulty satellite's impact on positioning accuracy
    """
    df, metadata = run_rinex_through_prx(input_for_test_tlse_2024)
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str)
    df_first_epoch = df[
        (
            df.time_of_reception_in_receiver_time
            == df.time_of_reception_in_receiver_time.min()
        )
        & (df.sat_elevation_deg > 10)
    ]
    constellation_to_use = ["G"]

    # observation with every sat
    obs = df_first_epoch[df.constellation.isin(constellation_to_use)]
    pt_lsq = spp_pt_lsq(obs)
    position_offset = pt_lsq[0:3, :] - np.array(
        metadata["approximate_receiver_ecef_position_m"]
    ).reshape(-1, 1)

    log.info(
        f"Using constellations: {constellation_to_use}, {len(obs.sv.unique())} SVs"
    )
    log.info(f"Position offset: {position_offset}")

    # observation without G27, using health_flag
    obs_without_G27 = obs.loc[obs.health_flag == 0]  # keep only healthy satellites
    pt_lsq_without_G27 = spp_pt_lsq(obs_without_G27)
    position_offset_without_G27 = pt_lsq_without_G27[0:3, :] - np.array(
        metadata["approximate_receiver_ecef_position_m"]
    ).reshape(-1, 1)
    log.info(
        f"Using constellations: {constellation_to_use}, {len(obs.sv.unique())} SVs (healthy)"
    )
    log.info(f"Position offset: {position_offset_without_G27}")

    # Verification: the position error significantly decrease when excluding G27
    assert np.linalg.norm(position_offset_without_G27) < np.linalg.norm(position_offset)
    # Verification: the position error without G27 is within tolerance
    assert np.max(np.abs(position_offset_without_G27)) < 1e1


def test_spp_lsq_for_obs_file_across_two_days(
    input_for_test_with_first_epoch_at_midnight,
):
    df, metadata = run_rinex_through_prx(
        input_for_test_with_first_epoch_at_midnight["obs_file"],
        prx_level=2,
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


def test_prx_level_1(input_for_test_tlse):
    main.process(observation_file_path=input_for_test_tlse, prx_level=1)
    expected_prx_file = Path(str(input_for_test_tlse).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()

    # Read first line of file, containing meta-data
    metadata = parse_prx_csv_file_metadata(expected_prx_file)
    assert metadata["prx_level"] == 1

    # Read the CSV file
    df = pd.read_csv(expected_prx_file, comment="#")

    # Expected CSV column names
    expected_column_names = {
        "time_of_reception_in_receiver_time",
        "C_obs_m",
        "D_obs_hz",
        "L_obs_cycles",
        "S_obs_dBHz",
        "rnx_obs_identifier",
        "constellation",
        "prn",
        "carrier_frequency_hz",
        "frequency_slot",
        "health_flag",
        "LLI",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "sat_elevation_deg",
        "sat_azimuth_deg",
        "ephemeris_hash",
    }

    # Checking if all renamed parameters exist in the dataframe columns
    assert set(df.columns) == expected_column_names, (
        f"Additional columns in computed prx file: {set(df.columns).difference(expected_column_names)}"
    )


def test_prx_level_2(input_for_test_tlse):
    main.process(observation_file_path=input_for_test_tlse, prx_level=2)
    expected_prx_file = Path(str(input_for_test_tlse).replace("crx.gz", "csv"))
    assert expected_prx_file.exists()

    # Read first line of file, containing meta-data
    metadata = parse_prx_csv_file_metadata(expected_prx_file)
    assert metadata["prx_level"] == 2

    # Read the CSV file
    df = pd.read_csv(expected_prx_file, comment="#")

    # Expected CSV column names
    expected_column_names = {
        "time_of_reception_in_receiver_time",
        "C_obs_m",
        "D_obs_hz",
        "L_obs_cycles",
        "S_obs_dBHz",
        "rnx_obs_identifier",
        "constellation",
        "prn",
        "carrier_frequency_hz",
        "frequency_slot",
        "health_flag",
        "LLI",
        "sat_pos_x_m",
        "sat_pos_y_m",
        "sat_pos_z_m",
        "sat_vel_x_mps",
        "sat_vel_y_mps",
        "sat_vel_z_mps",
        "sat_clock_offset_m",
        "sat_clock_drift_mps",
        "sat_code_bias_m",
        "relativistic_clock_effect_m",
        "sagnac_effect_m",
        "tropo_delay_m",
        "sat_code_bias_m",
        "iono_delay_m",
        "sat_elevation_deg",
        "sat_azimuth_deg",
        "ephemeris_hash",
    }

    # Checking if all renamed parameters exist in the dataframe columns
    assert set(df.columns) == expected_column_names, (
        f"Additional columns in computed prx file: {set(df.columns).difference(expected_column_names)}"
    )


def test_prx_level_3(input_for_test_tlse):
    """
    Test is currently inactive. To be changed once PRX level 3 works.
    """
    # check that calling main.process with "prx_level=3" will raise an AssertionError
    with pytest.raises(AssertionError):
        main.process(observation_file_path=input_for_test_tlse, prx_level=3)


def test_bootstrap_coarse_receiver_position(input_for_test_tlse):
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
