from pathlib import Path
import polars as pl
from prx import constants, converters, util
from datetime import datetime, timedelta


def parse_bia_file(filepath_bia_gz: Path) -> pl.DataFrame:
    """
            Parse gzipped BIA file and returns a pl.DataFrame with columns:
            - sat_id:
            - obs_id: rinex obs identifier
            - sat_hw_bias_m: bias value in meters
            - start: timestamp
            - end: timestamp

            Example:
    ┌────────┬────────┬───────────────┬─────────────────────┬─────────────────────┐
    │ sat_id ┆ obs_id ┆ sat_hw_bias_m ┆ start               ┆ end                 │
    │ ---    ┆ ---    ┆ ---           ┆ ---                 ┆ ---                 │
    │ str    ┆ str    ┆ f64           ┆ datetime[ns]        ┆ datetime[ns]        │
    ╞════════╪════════╪═══════════════╪═════════════════════╪═════════════════════╡
    │ G01    ┆ C1C    ┆ -0.425645     ┆ 2023-01-01 00:00:00 ┆ 2023-01-02 00:00:00 │
    │ G01    ┆ C1W    ┆ -0.0          ┆ 2023-01-01 00:00:00 ┆ 2023-01-02 00:00:00 │
    │ …      ┆ …      ┆ …             ┆ …                   ┆ …                   │
    │ E36    ┆ L5Q    ┆ -0.018269     ┆ 2023-01-01 00:00:00 ┆ 2023-01-02 00:00:00 │
    │ E36    ┆ L5X    ┆ -0.018269     ┆ 2023-01-01 00:00:00 ┆ 2023-01-02 00:00:00 │
    └────────┴────────┴───────────────┴─────────────────────┴─────────────────────┘

    """

    @util.disk_cache.cache(ignore=["filepath_bia_gz"])
    def cached_load(filepath_bia_gz: Path, file_hash: str):
        filepath_bia = converters.compressed_to_uncompressed(filepath_bia_gz)
        with open(filepath_bia, "r", encoding="cp1250") as f:
            sat_id_list = []
            obs1_list = []
            val_list = []
            start_list = []
            end_list = []
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
                    start = datetime(int(line[35:39]), 1, 1) + timedelta(
                        days=int(line[40:43]) - 1, seconds=int(line[44:49])
                    )
                    end = datetime(int(line[50:54]), 1, 1) + timedelta(
                        days=int(line[55:58]) - 1, seconds=int(line[59:64])
                    )
                    sat_id_list.append(sat_id)
                    obs1_list.append(obs1)
                    val_list.append(estimated_value)
                    start_list.append(start)
                    end_list.append(end)
            bia_df = pl.DataFrame(
                {
                    "sat_id": sat_id_list,
                    "obs_id": obs1_list,
                    "sat_hw_bias_m": val_list,
                    "start": start_list,
                    "end": end_list,
                },
            ).with_columns(
                pl.col("start").cast(pl.Datetime("ns")),
                pl.col("end").cast(pl.Datetime("ns")),
            )
            return bia_df

    file_content_hash = util.hash_of_file_content(filepath_bia_gz)
    return cached_load(filepath_bia_gz, file_content_hash)
