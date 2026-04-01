import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest

from prx.precise_corrections.antex import antex_processing


@pytest.fixture
def input_for_test():
    test_directory = Path(f"./tmp_test_directory_{__name__}").resolve()
    if test_directory.exists():
        # Make sure the expected file has not been generated before and is still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory)
    os.makedirs(test_directory)
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    test_file = {"obs": test_directory.joinpath(compressed_compact_rinex_file)}
    shutil.copy(
        Path(__file__).parent
        / f"datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_file["obs"],
    )
    assert test_file["obs"].exists()

    # Also provide ephemerides so the test does not have to download them:
    ephemerides_file = "BRDC00IGS_R_20230010000_01D_MN.rnx.zip"
    test_file["nav"] = test_directory.joinpath(ephemerides_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{ephemerides_file}",
        test_file["nav"].parent.joinpath(ephemerides_file),
    )
    assert test_file["nav"].parent.joinpath(ephemerides_file).exists()

    # sp3 file
    sp3_file = "GFZ0MGXRAP_20230010000_01D_05M_ORB.SP3"
    test_file["sp3"] = test_directory.joinpath(sp3_file)
    shutil.copy(
        Path(__file__).parent / f"datasets/TLSE_2023001/{sp3_file}",
        test_file["sp3"].parent.joinpath(sp3_file),
    )
    assert test_file["sp3"].parent.joinpath(sp3_file).exists()

    yield test_file
    shutil.rmtree(test_directory)


def test_pco_sat():
    # sat frame: z pointing to the geocenter, y perpendicular to the plane (geocenter,sun,sat), x = cross(e_y,e_z)

    # First scenario:                                           (sun)
    #   - sat, sun, geocenter at z_ecef = 0
    #   - sun at [0, y_ecef = 1 AU, 0]                                         ↑ y_ecef
    #   - sat at [x_ecef = a_sat, 0, 0]                        (earth)  (sat)  |--> x_ecef
    #   ==> the 3 axis of the sat frame are approximately x_sat = +y_ecef, y_sat = -z_ecef, z_sat = -x_ecef
    # R_sat2ecef = [ 0  0 -1
    #               +1  0  0
    #                0 -1  0]
    # With a pco of [1,0,1] in sat frame, we should obtain a pco of [-1,1,0] in ecef frame
    sun_pos_ecef = np.array([0, 149_597_870_700, 0])
    sat_pos_ecef = np.array([20_200 + 6_400, 0, 0])
    rx_pos_ecef = np.array([6_400, 0, 0])
    pco_sat = np.array([1, 0, 1]).reshape(-1, 1)

    e_z = -sat_pos_ecef / np.linalg.norm(sat_pos_ecef)
    e_s = (sun_pos_ecef - rx_pos_ecef) / np.linalg.norm(sun_pos_ecef - rx_pos_ecef)
    e_y = np.cross(e_z, e_s) / np.linalg.norm(np.cross(e_z, e_s))
    e_x = np.cross(e_y, e_z)

    rot_sat2mat = np.stack([e_x, e_y, e_z]).T
    assert (rot_sat2mat == np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])).all()
    pco_ecef_1 = (rot_sat2mat @ pco_sat).reshape(3)
    assert (pco_ecef_1 == np.array([-1, 1, 0])).all()

    # Second scenario:                                         (earth)  (sat)
    #   - sat, sun, geocenter at z_ecef = 0
    #   - sun at [0, y_ecef = -1 AU, 0]                                  ↑ y_ecef
    #   - sat at [x_ecef = a_sat, 0, 0]                         (sun)    |--> x_ecef
    #   ==> the 3 axis of the sat frame are approximately x_sat = -y_ecef, y_sat = +z_ecef, z_sat = -x_ecef
    # R_sat2ecef = [ 0  0 -1
    #               -1  0  0
    #                0 +1  0]
    # With a pco of [1,0,1] in sat frame, we should obtain a pco of [-1,-1,0] in ecef frame
    sun_pos_ecef = np.array([0, -149_597_870_700, 0])
    sat_pos_ecef = np.array([20_200 + 6_400, 0, 0])
    rx_pos_ecef = np.array([6_400, 0, 0])
    pco_sat = np.array([1, 0, 1]).reshape(-1, 1)

    e_z = -sat_pos_ecef / np.linalg.norm(sat_pos_ecef)
    e_s = (sun_pos_ecef - rx_pos_ecef) / np.linalg.norm(sun_pos_ecef - rx_pos_ecef)
    e_y = np.cross(e_z, e_s) / np.linalg.norm(np.cross(e_z, e_s))
    e_x = np.cross(e_y, e_z)

    rot_sat2mat = np.stack([e_x, e_y, e_z]).T
    assert (rot_sat2mat == np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])).all()
    pco_ecef_2 = (rot_sat2mat @ pco_sat).reshape(3)
    assert (pco_ecef_2 == np.array([-1, -1, 0])).all()

    # pco computation with function
    timestamp1 = pd.Timestamp("2026-03-20 06:00:00")  # sun "close" to [0,1,0]
    timestamp2 = pd.Timestamp("2026-03-20 18:00:00")  # sun "close" to [0,-1,0]
    query = pd.DataFrame(
        {
            "sv": np.array(["G01"] * 2),
            "query_time_isagpst": np.array([timestamp1, timestamp2]),
            "signal": "C1C",
        }
    )
    query[["sat_pos_com_x_m", "sat_pos_com_y_m", "sat_pos_com_z_m"]] = np.array(
        [sat_pos_ecef] * 2
    )
    pco_function = antex_processing.compute_pco_sat(
        query,
        pd.DataFrame(
            data={
                "satellite_or_serial_no": ["G01"] * 5,
                "valid_from": [pd.Timestamp("2020-01-01")] * 5,
                "valid_until": [pd.Timestamp("2030-01-01")] * 5,
                "constellation": ["G"] * 5,
                "carrier_freq_id": [1, 2, 5, 6, 7],
                "pco_north_m": [1] * 5,
                "pco_east_m": [0] * 5,
                "pco_up_m": [1] * 5,
            },
        ),
    )

    # tolerance on pco computation, mainly due to imprecise date leading to slightly different sun position
    tol = 1e-2
    assert pco_function.loc[
        pco_function.query_time_isagpst == timestamp1,
        ["pco_sat_x_m", "pco_sat_y_m", "pco_sat_z_m"],
    ].to_numpy() == pytest.approx(np.array([pco_ecef_1]), abs=tol)
    assert pco_function.loc[
        pco_function.query_time_isagpst == timestamp2,
        ["pco_sat_x_m", "pco_sat_y_m", "pco_sat_z_m"],
    ].to_numpy() == pytest.approx(np.array([pco_ecef_2]), abs=tol)
