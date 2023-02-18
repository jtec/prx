from pathlib import Path
import glob
import itertools
import subprocess
import gzip
import prx
import helpers


log = helpers.get_logger(__name__)


def compressed_to_uncompressed(file: Path):
    assert file.exists(), "File does not exist"
    if str(file).endswith(".gz"):
        uncompressed_file = Path(str(file).replace(".gz", ""))
        with gzip.open(file, "rb") as compressed_file:
            with open(uncompressed_file, "wb") as output_file:
                output_file.write(compressed_file.read())
        log.info(f"Uncompressed {file} to {uncompressed_file}")
        return uncompressed_file
    else:
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
        str(prx.prx_root().joinpath("tools/RNXCMP/**/CRX2RNX*")), recursive=True
    )
    for crx2rnx_binary in crx2rnx_binaries:
        command = f" {crx2rnx_binary} {file}"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            expanded_file = Path(str(file).replace(".crx", ".rnx"))
            log.info(f"Converted compact Rinex to Rinex: {expanded_file}")
            return expanded_file
    return None


def rinex_2_to_rinex_3(file: Path):
    assert file.exists(), "File does not exist"
    return None


def is_rinex_3_obs_file(file: Path):
    assert file.exists(), "File does not exist"
    try:
        with open(file) as f:
            first_line = f.readline()
    except UnicodeDecodeError as e_unicode:
        return False
    if "RINEX VERSION" not in first_line or "3.0" not in first_line:
        return False
    return True


def anything_to_rinex_3(file: Path):
    assert file.exists(), "File does not exist"
    file = Path(file)
    converters = [
        compact_rinex_obs_file_to_rinex_obs_file,
        compressed_to_uncompressed,
        rinex_2_to_rinex_3,
    ]
    output = None
    input = file
    converter_calls = 0
    max_number_of_conversions = 10
    for converter in itertools.cycle(converters):
        if is_rinex_3_obs_file(input):
            return input
        converter_calls += 1
        output = converter(input)
        if output is not None:
            input = output
        if converter_calls > max_number_of_conversions * len(converters):
            log.error(
                f"Tried converting file {file.name} {max_number_of_conversions} times, still not RINEX 3, giving up."
            )
            return None
