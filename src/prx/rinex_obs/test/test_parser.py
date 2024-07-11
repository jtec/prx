from pathlib import Path
from prx.rinex_obs.parser import parse as prx_obs_parse
import numpy as np

from prx import helpers, converters


def test_compare_to_georinex():
    file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx_slice_0.24h.rnx.gz"
    )
    georinex_output = helpers.parse_rinex_obs_file(file)
    prx_output = prx_obs_parse(file)
    merged = prx_output.merge(
        georinex_output, on=["time", "sv", "obs_type"], suffixes=("_prx", "_georinex")
    )
    merged["obs_value_diff"] = merged.obs_value_georinex - merged.obs_value_prx
    assert np.isclose(merged.obs_value_diff.abs().max(), 0)
