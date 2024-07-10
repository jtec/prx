from pathlib import Path

import numpy as np

from prx import helpers


def test_compare_to_georinex():
    file = (
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx_slice_0.24h.rnx"
    )
    georinex_output = helpers.parse_rinex_obs_file(file, "georinex").dropna()
    prx_output = helpers.parse_rinex_obs_file(file, "prx").dropna()
    merged = prx_output.merge(
        georinex_output, on=["time", "sv", "obs_type"], suffixes=("_prx", "_georinex")
    )
    merged["obs_value_diff"] = merged.obs_value_georinex - merged.obs_value_prx
    assert np.isclose(merged.obs_value_diff.abs().max(), 0)
