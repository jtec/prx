from pathlib import Path
from prx import helpers
from prx.converters import anything_to_rinex_3
from prx.user import parse_prx_csv_file
from prx.main import process as prx_process

log = helpers.get_logger(__name__)


def parse(obs_file: Path):
    rnx_file = anything_to_rinex_3(obs_file)
    assert rnx_file, f"Failed to convert {obs_file} to RINEX 3"
    if not rnx_file.exists():
        prx_process(rnx_file)
    df, metadata = parse_prx_csv_file(rnx_file.with_suffix(".csv"))
    # Receiver clocks are off by at most a few tens of milliseconds, so let's round for matching rover-base obs
    df["time_of_reception_rounded"] = df.time_of_reception_in_receiver_time.round(
        "50ms"
    )
    df["sv"] = df["constellation"].astype(str) + df["prn"].astype(str).str.pad(
        2, fillchar="0"
    )
    return df


# Provides and example of how to compute between-receivers single
# differences and double differences after processing obs files with prx.
def main():
    # We'll use two IGS stations that are just roughly a kilometer apart
    df_rover, df_base = (
        parse(Path(__file__).parent / "obs/TLSE00FRA_R_20241900000_15M_01S_MO.crx.gz"),
        parse(Path(__file__).parent / "obs/TLSG00FRA_R_20241900000_15M_01S_MO.crx.gz"),
    )
    df = df_rover.merge(
        df_base,
        on=["time_of_reception_rounded", "sv", "rnx_obs_identifier"],
        suffixes=("_rover", "_base"),
    ).reset_index()
    df["C_obs_m_sd"] = df.C_obs_m_rover + df.C_obs_m_base
    df["L_obs_cycles_sd"] = df.L_obs_cycles_rover + df.L_obs_cycles_base
    df["ref_sv"] = ""

    print(df.info())


if __name__ == "__main__":
    main()
