import shutil
from pathlib import Path
import glob
import itertools
import subprocess
import gzip
import gzinfo
import logging
import prx

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

def compressed_to_uncompressed(file: Path):
    if str(file).endswith(".gz"):
        uncompressed_file = str(file).replace(".gz", "")
        with gzip.open(file, 'rb') as compressed_file:
            with open(uncompressed_file, 'wb') as output_file:
                output_file.write(compressed_file.read())
        logger.info(f"Uncompressed file(s) to {uncompressed_file}")
        return uncompressed_file
    else:
        return None


def compact_rinex_to_rinex(file: Path):
    if not str(file).endswith(".crx"):
        return None
    with open(file) as f:
        first_line = f.readline()
    if "COMPACT RINEX FORMAT" not in first_line:
        return None
    crx2rnx_binaries = glob.glob(str(prx.prx_root().joinpath("tools/RNXCMP/**/CRX2RNX*")), recursive=True)
    for crx2rnx_binary in crx2rnx_binaries:
        command = f" {crx2rnx_binary} {file}"
        result = subprocess.run(command, capture_output=True, shell=True)
        if result.returncode == 0:
            logger.info("Converted compact Rinex to Rinex: {}")


def rinex_2_to_rinex_3(file: Path):
    return None

def is_rinex_3(file: Path):
    return False

def anything_to_rinex_3(file: Path):
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
        if is_rinex_3(output):
            break
        if output is not None:
            input = output
