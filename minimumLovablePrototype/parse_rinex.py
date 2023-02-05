import os

import georinex
import helpers
from pathlib import Path
import xarray

log = helpers.get_logger(__name__)


# Can speed up RINEX parsing by using parsing results previously obtained and saved to disk.
def load(rinex_file: Path, use_caching=False):
    cache_directory = Path(__file__).resolve().parent.joinpath("afterburner")
    os.makedirs(cache_directory, exist_ok=True)
    if use_caching:
        cache_file = cache_directory.joinpath(f"{rinex_file.stem}_{helpers.md5(rinex_file)}.nc")
        if cache_file.exists():
            log.info(f"Loading cached parsing result for {rinex_file} from {cache_file}")
            return xarray.open_dataset(cache_file)
    parsed = georinex.load(rinex_file)
    if use_caching:
        parsed.to_netcdf(cache_file)
    return parsed
