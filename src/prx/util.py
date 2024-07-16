from pathlib import Path


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
