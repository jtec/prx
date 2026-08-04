from pathlib import Path
import polars as pl
from prx import constants, converters


def parse_bia_file(filepath_bia_gz: Path) -> pl.DataFrame:
    """
    Parse gzipped BIA file and returns a pl.DataFrame with columns:
    - sat_id
    - one column per obs identifier (e.g. "C1C", "L2X", etc), containing the bias value in meters

    Example:
    ┌────────┬──────────┬──────────┬─────────┬───┬──────────┬──────────┬──────────┬──────────┐
    │ sat_id ┆ C1C      ┆ C1W      ┆ C2L     ┆ … ┆ L2X      ┆ L2W      ┆ L5Q      ┆ L5X      │
    │ ---    ┆ ---      ┆ ---      ┆ ---     ┆   ┆ ---      ┆ ---      ┆ ---      ┆ ---      │
    │ str    ┆ f64      ┆ f64      ┆ f64     ┆   ┆ f64      ┆ f64      ┆ f64      ┆ f64      │
    ╞════════╪══════════╪══════════╪═════════╪═══╪══════════╪══════════╪══════════╪══════════╡
    │ G01    ┆ 9.2018   ┆ 10.5943  ┆ 16.1153 ┆ … ┆ 0.22928  ┆ 0.22928  ┆ null     ┆ null     │
    │ G02    ┆ -10.2757 ┆ -11.9688 ┆ null    ┆ … ┆ 0.70426  ┆ 0.70426  ┆ null     ┆ null     │
    │ G03    ┆ 6.7032   ┆ 7.3863   ┆ 12.852  ┆ … ┆ 0.69414  ┆ 0.69414  ┆ null     ┆ null     │
    │ …      ┆ …        ┆ …        ┆ …       ┆ … ┆ …        ┆ …        ┆ …        ┆ …        │
    └────────┴──────────┴──────────┴─────────┴───┴──────────┴──────────┴──────────┴──────────┘

    """
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
                assert unit == "ns", f"Wrong unit in file. Expected 'ns', read '{unit}'"
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
            {"sat_id": sat_id_list, "obs1": obs1_list, "val": val_list}
        ).pivot(on="obs1", index="sat_id", values="val")
        return bia_df
