from pathlib import Path
import parse_rinex_observations as obs_sparser


def test_resources_directory():
    return Path(__file__).parent.joinpath("test_resources")


def test_parse_successfully():
    # When parsing a valid RINEX 3 observation file
    test_file = test_resources_directory().joinpath("valid_rinex_3.rnx")
    parsed = obs_sparser.parse_rinex_3_obs(test_file)
    # ... we expect it to be successfully parsed
    assert(parsed is not None)
