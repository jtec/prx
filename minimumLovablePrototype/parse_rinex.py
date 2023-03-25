import os
import georinex
from pathlib import Path
import xarray
import pickle

import helpers

log = helpers.get_logger(__name__)


# Can speed up RINEX parsing by using parsing results previously obtained and saved to disk.


def load(rinex_file: Path, use_caching=False):
    return load_from_pickle_cache(rinex_file, use_caching)


def load_from_pickle_cache(rinex_file: Path, use_caching=False):
    cache_directory = Path(__file__).resolve().parent.joinpath("afterburner")
    os.makedirs(cache_directory, exist_ok=True)
    if use_caching:
        cache_file = cache_directory.joinpath(
            f"{rinex_file.stem}_{helpers.md5_of_file_content(rinex_file)}.pickle"
        )
        if cache_file.exists():
            log.info(
                f"Loading cached parsing result for {rinex_file} from {cache_file}"
            )
            try:
                with open(cache_file, "rb") as file:
                    cache = pickle.load(file)
                return cache
            except Exception as e:
                log.error(
                    f"Exception when loading cached pickle, removing cache file and parsing. Exception: {e}"
                )
                os.remove(cache_file)
                return load_from_pickle_cache(rinex_file, use_caching)
    log.info(f"Parsing {rinex_file} ...")
    parsed = georinex.load(rinex_file)
    if use_caching:
        try:
            with open(cache_file, "wb") as file:
                pickle.dump(parsed, file)
        except Exception as e:
            log.error(
                f"Exception when writing parsed RINEX to pickle cache. Moving on without saving. Exception: {e}"
            )
    return parsed


def load_from_netcdf_cache(rinex_file: Path, use_caching=False):
    cache_directory = Path(__file__).resolve().parent.joinpath("afterburner")
    os.makedirs(cache_directory, exist_ok=True)
    if use_caching:
        cache_file = cache_directory.joinpath(
            f"{rinex_file.stem}_{helpers.md5_of_file_content(rinex_file)}.nc"
        )
        if cache_file.exists():
            log.info(
                f"Loading cached parsing result for {rinex_file} from {cache_file}"
            )
            try:
                cache = xarray.open_dataset(cache_file)
                return cache
            except Exception as e:
                log.error(
                    f"Exception when loading cached NetCDF data, removing cache file and parsing. Exception: {e}"
                )
                os.remove(cache_file)
                return load_from_netcdf_cache(rinex_file, use_caching)
    parsed = georinex.load(rinex_file)
    if use_caching:
        try:
            parsed.to_netcdf(cache_file)
        except Exception as e:
            log.error(
                f"Exception when writing parsed RINEX to NetCDF cache. moving on without saving. Exception: {e}"
            )
    return parsed
