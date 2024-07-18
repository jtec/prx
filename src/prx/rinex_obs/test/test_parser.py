from pathlib import Path
import georinex
from prx.helpers import repair_with_gfzrnx, obs_dataset_to_obs_dataframe
from prx.rinex_obs.parser import parse as prx_obs_parse

from prx import converters


def test_compare_to_georinex():
    file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx_slice_0.24h.rnx.gz"
    )
    file = Path(
        "/Users/janbolting/repositories/prx/src/demos/rover_base_differencing/obs/TLSE00FRA_R_20241900000_15M_01S_MO.rnx")
    repair_with_gfzrnx(file)
    prx_output = (
        prx_obs_parse(file)
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    ...
    georinex_output = (
        obs_dataset_to_obs_dataframe(georinex.load(file))
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    assert georinex_output.equals(prx_output)
