import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest

from prx import util

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
    pco_corr_list = []
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
    pco_ecef = (rot_sat2mat @ pco_sat).reshape(3)
    assert (pco_ecef == np.array([-1, 1, 0])).all()
    pco_corr_list.append(pco_ecef.dot(rx_pos_ecef) / np.linalg.norm(rx_pos_ecef))

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
    pco_ecef = (rot_sat2mat @ pco_sat).reshape(3)
    assert (pco_ecef == np.array([-1, -1, 0])).all()
    pco_corr_list.append(pco_ecef.dot(rx_pos_ecef) / np.linalg.norm(rx_pos_ecef))

    # pco correction computation with function
    pco_corr_function, _ = util.compute_pco_sat(
        np.array(["G01"] * 2),
        np.array([sat_pos_ecef] * 2),
        np.array([rx_pos_ecef] * 2),
        np.array(
            [
                pd.Timestamp(
                    year=2020, month=6, day=20, hour=6, minute=6
                ),  # approx time when the sun is "close" to [0,1,0]
                pd.Timestamp(
                    year=2020, month=6, day=20, hour=18, minute=6
                ),  # approx time when the sun is "close" to [0,-1,0]
            ]
        ),
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
    assert pco_corr_list == pytest.approx(pco_corr_function[:, 0])

