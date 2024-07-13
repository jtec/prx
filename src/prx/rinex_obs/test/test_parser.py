from pathlib import Path

from prx.rinex_obs.parser import parse as prx_obs_parse

from prx import converters, helpers


def test_compare_to_georinex():
    file = converters.anything_to_rinex_3(
        Path(__file__).parent
        / "datasets"
        / "TLSE00FRA_R_20220010000_01D_30S_MO.rnx_slice_0.24h.rnx.gz"
    )
    georinex_output = (
        helpers.parse_rinex_obs_file(file)
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    prx_output = (
        prx_obs_parse(file)
        .sort_values(by=["time", "sv", "obs_type"])
        .reset_index(drop=True)
    )
    assert georinex_output.equals(prx_output)
