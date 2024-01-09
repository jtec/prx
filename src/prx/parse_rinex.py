import georinex
from pathlib import Path

import pandas as pd
import joblib
from prx import helpers

log = helpers.get_logger(__name__)

memory = joblib.Memory(Path(__file__).parent.joinpath("diskcache"), verbose=0)


# Can speed up RINEX parsing by using parsing results previously obtained and saved to disk.
def load(rinex_file: Path):
    @memory.cache
    def cached_load(rinex_file: Path, file_hash: str):
        log.info(f"Parsing {rinex_file} ...")
        helpers.repair_with_gfzrnx(rinex_file)
        parsed = georinex.load(rinex_file)
        return parsed

    t0 = pd.Timestamp.now()
    file_content_hash = helpers.hash_of_file_content(rinex_file)
    hash_time = pd.Timestamp.now() - t0
    if hash_time > pd.Timedelta(seconds=1):
        log.info(
            f"Hashing file content took {hash_time}, we might want to partially hash the file"
        )
    return cached_load(rinex_file, file_content_hash)
