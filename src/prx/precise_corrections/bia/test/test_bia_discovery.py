import shutil
from pathlib import Path

import pandas as pd
import pytest

from prx import util
from prx.precise_corrections.sp3.sp3_file_discovery import sp3_file_folder

from prx.precise_corrections.bia import bia_file_discovery as discovery


@pytest.fixture
def set_up_test(tmp_path_factory):
    test_sp3_file = (
        sp3_file_folder(pd.Timestamp("2023-01-01"))
        / "COD0MGXFIN_20230010000_01D_05M_ORB.SP3.gz"
    )
    shutil.copy(
        util.prx_src_directory().joinpath(
            "test",
            "datasets",
            "TLSE_2023001",
            "COD0MGXFIN_20230010000_01D_05M_ORB.SP3.gz",
        ),
        test_sp3_file,
    )
    assert test_sp3_file.exists()

    yield {
        "year": 2023,
        "doy": 1,
        "analysis_center": "COD",
        "sp3": test_sp3_file,
    }


def test_bia_file_online_availability(set_up_test):
    year = set_up_test["year"]
    doy = set_up_test["doy"]
    ac = set_up_test["analysis_center"]
    assert discovery.check_online_availability(
        year, doy, ac
    ) == discovery.bia_file_folder(year, doy) / discovery.build_bia_file_name(
        year, doy, ac
    )


def test_download_bia_file():
    # delete local file to trigger download
    path_local = discovery.bia_file_folder(2023, 1) / discovery.build_bia_file_name(
        2023, 1, "COD"
    )
    path_local.unlink(missing_ok=True)
    file = discovery.discover_or_download_bia_file(
        Path("COD0MGXFIN_20230010000_01D_05M_ORB.SP3.gz")
    )
    assert file.exists()


def test_find_local_bia_file(set_up_test):
    file = discovery.discover_or_download_bia_file(set_up_test["sp3"])
    assert file.exists()
