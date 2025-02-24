import logging
import platform
import re
import subprocess
from functools import wraps
from pathlib import Path

import pandas as pd
from prx import constants

logger = logging.getLogger(__name__)


def file_exists_and_can_read_first_line(file: Path):
    assert file.exists(), f"Provided file path {file} does not exist"
    try:
        with open(file) as f:
            return f.readline()
    except UnicodeDecodeError:
        return None


def is_rinex_3_obs_file(file: Path):
    first_line = file_exists_and_can_read_first_line(file)
    if first_line is None:
        return False
    if "RINEX VERSION" not in first_line or "3.0" not in first_line:
        return False
    if "NAVIGATION DATA" in first_line:
        return False
    return True


def is_rinex_3_nav_file(file: Path):
    first_line = file_exists_and_can_read_first_line(file)
    if first_line is None:
        return False
    if "NAVIGATION DATA" not in first_line or "3.0" not in first_line:
        return False
    return True


def is_rinex_2_obs_file(file: Path):
    first_line = file_exists_and_can_read_first_line(file)
    if first_line is None:
        return False
    if "RINEX VERSION" not in first_line or "2.0" not in first_line:
        return False
    return True


def is_rinex_2_nav_file(file: Path):
    first_line = file_exists_and_can_read_first_line(file)
    if first_line is None:
        return False
    if "NAV" not in first_line or "2." not in first_line:
        return False
    return True


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = pd.Timestamp.now()
        result = func(*args, **kwargs)
        end_time = pd.Timestamp.now()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} took {total_time} to run.")
        return result

    return timeit_wrapper


@timeit
def repair_with_gfzrnx(file):
    with open(file) as f:
        if "gfzrnx" in f.read():
            logging.warning(f"File {file} already contains 'gfzrnx', skipping repair.")
            return file
    path_folder_gfzrnx = Path(__file__).parent.joinpath("tools", "gfzrnx")
    path_binary = path_folder_gfzrnx.joinpath(
        constants.gfzrnx_binary[platform.system()]
    )
    command = [
        str(path_binary),
        "-finp",
        str(file),
        "-fout",
        str(file),
        "-chk",
        "-kv",
        "-f",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info(f"Ran gfzrnx file repair on {file}")
        with open(file, "r") as f:
            file_content = f.read()
            file_content = re.sub(
                r"gfzrnx-(.*?)FILE PROCESSING(.*)UTC COMMENT",
                r"gfzrnx-\1FILE PROCESSING (timestamp removed by prx) UTC COMMENT",
                file_content,
                count=1,
            )
        with open(file, "w") as f:
            f.write(file_content)
            logger.info(
                f"Removed repair timestamp from gfzrnx file {file} to avoid content hash changes."
            )
    else:
        logger.info(f"gfzrnx file repair run failed: {result}")
        assert False
    return file
