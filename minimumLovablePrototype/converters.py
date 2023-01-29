from pathlib import Path
import glob
import itertools
import subprocess
import gzip
import logging
import prx

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)


def compressed_to_uncompressed(file: Path):
    assert file.exists()
    if str(file).endswith(".gz"):
        uncompressed_file = Path(str(file).replace(".gz", ""))
        with gzip.open(file, 'rb') as compressed_file:
            with open(uncompressed_file, 'wb') as output_file:
                output_file.write(compressed_file.read())
        logger.info(f"Uncompressed {file} to {uncompressed_file}")
        return uncompressed_file
    else:
        return None

def is_compact_rinex(file: Path):
    assert file.exists()
    if not str(file).endswith(".crx"):
        return False
    with open(file) as f:
        first_line = f.readline()
    if "COMPACT RINEX FORMAT" not in first_line:
        return False
    return True


def compact_rinex_to_rinex(file: Path):
    assert file.exists()
    if not is_compact_rinex(file):
        return None
    crx2rnx_binaries = glob.glob(str(prx.prx_root().joinpath("tools/RNXCMP/**/CRX2RNX*")), recursive=True)
    for crx2rnx_binary in crx2rnx_binaries:
        command = f" {crx2rnx_binary} {file}"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            expanded_file = Path(str(file).replace(".crx", ".rnx"))
            logger.info(f"Converted compact Rinex to Rinex: {expanded_file}")
            return expanded_file
    return None


def rinex_2_to_rinex_3(file: Path):
    assert file.exists()
    return None

def is_rinex_3(file: Path):
    assert file.exists()
    with open(file) as f:
        first_line = f.readline()
    if "RINEX VERSION" not in first_line or "3.0" not in first_line:
        return False
    return True

def anything_to_rinex_3(file: Path):
    assert file.exists()
    file = Path(file)
    converters = [
        compact_rinex_to_rinex,
        compressed_to_uncompressed,
        rinex_2_to_rinex_3,
    ]
    output = None
    input = file
    for converter in itertools.cycle(converters):
        output = converter(input)
        if output is not None:
            if is_rinex_3(output):
                return output
            input = output
    return None