from pathlib import Path
import glob
import itertools
import subprocess
import gzip
import zipfile
from prx.helpers import get_logger
from prx.util import (
    is_rinex_2_obs_file,
    is_rinex_2_nav_file,
    is_rinex_3_obs_file,
    is_rinex_3_nav_file,
)

log = get_logger(__name__)


def compressed_to_uncompressed(file: Path):
    assert file.exists(), "File does not exist"
    if str(file).endswith(".gz"):
        uncompressed_file = Path(str(file).replace(".gz", ""))
        with gzip.open(file, "rb") as compressed_file:
            with open(uncompressed_file, "wb") as output_file:
                output_file.write(compressed_file.read())
        log.info(f"Uncompressed {file} to {uncompressed_file}")
        return uncompressed_file
    if str(file).endswith(".zip"):
        with zipfile.ZipFile(file, mode="r") as archive:
            assert (
                len(archive.namelist()) == 1
            ), "Not expecting more than one file in archive here."
            uncompressed_file = file.parent.joinpath(archive.namelist()[0])
            archive.extract(uncompressed_file.name, uncompressed_file.parent)
        log.info(f"Uncompressed {file} to {uncompressed_file}")
        return uncompressed_file
    return None


def is_compact_rinex_obs_file(file: Path):
    assert file.exists(), "File does not exist"
    if not str(file).endswith(".crx"):
        return False
    with open(file) as f:
        first_line = f.readline()
    if "COMPACT RINEX FORMAT" not in first_line:
        return False
    return True


def compact_rinex_obs_file_to_rinex_obs_file(file: Path):
    assert file.exists(), "File does not exist"
    if not is_compact_rinex_obs_file(file):
        return None
    crx2rnx_binaries = glob.glob(
        str(Path(__file__).parent / "tools/RNXCMP/**/CRX2RNX*"), recursive=True
    )
    assert len(crx2rnx_binaries) > 0, "Could not find any CRX2RNX binary"
    for crx2rnx_binary in crx2rnx_binaries:
        command = f" {crx2rnx_binary} -f {file}"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            expanded_file = Path(str(file).replace(".crx", ".rnx"))
            log.info(f"Converted compact Rinex to Rinex: {expanded_file}")
            return expanded_file
    return None


def rinex_2_to_rinex_3(file: Path):
    if is_rinex_2_obs_file(file) or is_rinex_2_nav_file(file):
        log.debug(f"RINEX 2 not supported: {file}")
        return None


def anything_to_rinex_3(file: Path):
    assert file.exists(), "File does not exist"
    file = Path(file)
    converters = [
        compact_rinex_obs_file_to_rinex_obs_file,
        compressed_to_uncompressed,
        rinex_2_to_rinex_3,
    ]
    input = file
    max_number_of_cycles = 10
    for converter_calls, converter in enumerate(itertools.cycle(converters)):
        if is_rinex_3_obs_file(input) or is_rinex_3_nav_file(input):
            return input
        output = converter(input)
        if output is not None:
            input = output
        if converter_calls > max_number_of_cycles * len(converters):
            log.error(
                f"Tried converting file {file.name} {max_number_of_cycles} times, still not RINEX 3, giving up."
            )
            return None
