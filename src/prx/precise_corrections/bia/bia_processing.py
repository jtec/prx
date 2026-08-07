from pathlib import Path
import polars as pl
from prx import constants, converters, util


def parse_bia_file(filepath_bia_gz: Path) -> pl.DataFrame:
    """
        Parse gzipped BIA file and returns a pl.DataFrame with columns:
        - sat_id:
        - obs_id: rinex obs identifier
        - sat_hw_bias_m: bias value in meters

        Example:
    ┌────────┬────────┬───────────────┐
    │ sat_id ┆ obs_id ┆ sat_hw_bias_m │
    │ ---    ┆ ---    ┆ ---           │
    │ str    ┆ str    ┆ f64           │
    ╞════════╪════════╪═══════════════╡
    │ G01    ┆ C1C    ┆ 2.75863       │
    │ G01    ┆ C1W    ┆ 3.176091      │
    │ G01    ┆ C2L    ┆ 4.831245      │
    │ …      ┆ …      ┆ …             │
    │ E36    ┆ L1C    ┆ 0.074504      │
    │ E36    ┆ L1X    ┆ 0.074504      │
    │ E36    ┆ L5Q    ┆ 0.136439      │
    │ E36    ┆ L5X    ┆ 0.136439      │
    └────────┴────────┴───────────────┘

    """

    @util.disk_cache.cache(ignore=["filepath_bia_gz"])
    def cached_load(filepath_bia_gz: Path, file_hash: str):
        filepath_bia = converters.compressed_to_uncompressed(filepath_bia_gz)
        with open(filepath_bia, "r") as f:
            sat_id_list = []
            obs1_list = []
            val_list = []
            # find beginning of block BIAS/SOLUTION
            for line in f:
                if line.startswith("+BIAS/SOLUTION"):
                    break

            for line in f:
                if line.startswith("-BIAS/SOLUTION"):  # escape loop
                    break
                if line.startswith("*BIAS"):  # skip header
                    continue
                bias_type = line[0:4].strip()
                station = line[15:24].strip()
                if (station == "") and (
                    bias_type == "OSB"
                ):  # keep only OSB and satellite biases
                    unit = line[64:69].strip()
                    assert unit == "ns", (
                        f"Wrong unit in file. Expected 'ns', read '{unit}'"
                    )
                    sat_id = line[11:14].strip()
                    obs1 = line[25:29].strip()
                    estimated_value = (
                        float(line[70:91])
                        / constants.cNanoSecondsPerSecond
                        * constants.cGpsSpeedOfLight_mps
                    )
                    sat_id_list.append(sat_id)
                    obs1_list.append(obs1)
                    val_list.append(estimated_value)
            bia_df = pl.DataFrame(
                {"sat_id": sat_id_list, "obs_id": obs1_list, "sat_hw_bias_m": val_list}
            )
            return bia_df

    file_content_hash = util.hash_of_file_content(filepath_bia_gz)
    return cached_load(filepath_bia_gz, file_content_hash)
