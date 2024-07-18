from pathlib import Path
import pandas as pd

from prx import helpers
from prx.main import process as prx_process
from prx.converters import anything_to_rinex_3

log = helpers.get_logger(__name__)


# Provides and example of how to compute between-receivers single
# differences and double differences after processing obs files with prx.
def main():
    # We'll use two IGS stations that are just roughly a kilometer apart
    rx_base_obs = anything_to_rinex_3(Path(__file__).parent / "obs/TLSE00FRA_R_20241900000_15M_01S_MO.crx.gz")
    rx_rover_obs = anything_to_rinex_3(Path(__file__).parent / "obs/TLSG00FRA_R_20241900000_15M_01S_MO.crx.gz")
    if not rx_base_obs.with_suffix(".csv").exists():
        prx_process(rx_base_obs)
    if not rx_rover_obs.with_suffix(".csv").exists():
        prx_process(rx_rover_obs)
    df_rover = pd.read_csv(rx_rover_obs.with_suffix(".csv"))
    df_base = pd.read_csv(rx_base_obs.with_suffix(".csv"))


if __name__ == "__main__":
    main()
