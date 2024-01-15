import os
from pathlib import Path
import shutil

from prx import helpers, converters


def test_directory() -> Path:
    return Path(f"./tmp_test_directory_{__name__}").resolve()


def prepare_test():
    if test_directory().exists():
        # Make sure the expected files have not been generated before and are still on disk due to e.g. a previous
        # test run having crashed:
        shutil.rmtree(test_directory())
    os.makedirs(test_directory())


def clean_up_after_test():
    shutil.rmtree(test_directory())


def test_compressed_crx_to_rnx():
    prepare_test()
    compressed_compact_rinex_file = "TLSE00FRA_R_20230010100_10S_01S_MO.crx.gz"
    shutil.copy(
        helpers.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2023001/{compressed_compact_rinex_file}",
        test_directory().joinpath(compressed_compact_rinex_file),
    )
    rinex_3_file = converters.anything_to_rinex_3(
        test_directory().joinpath(compressed_compact_rinex_file)
    )
    expected_uncompacted_rinex_3_file = test_directory().joinpath(
        compressed_compact_rinex_file.replace("crx.gz", "rnx")
    )
    assert rinex_3_file is not None
    assert rinex_3_file.exists()
    assert rinex_3_file == expected_uncompacted_rinex_3_file
    clean_up_after_test()


def test_converting_file_that_cannot_be_converted():
    prepare_test()
    # When trying to convert a file that cannot be converted into RINEX 3, expect the converter to return None
    does_not_contain_rinex_3 = "igs21906.sp3"
    assert (
        helpers.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2022001/{does_not_contain_rinex_3}"
    ).exists()
    shutil.copy(
        helpers.prx_repository_root()
        / f"src/prx/test/datasets/TLSE_2022001/{does_not_contain_rinex_3}",
        test_directory().joinpath(does_not_contain_rinex_3),
    )
    assert (
        converters.anything_to_rinex_3(
            test_directory().joinpath(does_not_contain_rinex_3)
        )
        is None
    )
    clean_up_after_test()
