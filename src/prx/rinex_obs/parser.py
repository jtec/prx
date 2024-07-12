import pandas as pd
import re


def get_obs_types(header):
    types = []
    marker = "SYS / # / OBS TYPES"
    lines = " " + " ".join(
        header[header.lines.str.contains(marker)].lines.to_list()
    ).replace(marker, " ")
    blocks = re.split(r"( C | E | J | R | I | G | S )", lines)
    blocks = [block.strip() for block in blocks if len(block.strip()) > 0]
    blocks = [
        block + " " + blocks[i + 1] for i, block in enumerate(blocks) if len(block) == 1
    ]
    for block in blocks:
        types.append(
            {
                "constellation": block.split()[0].strip(),
                "number_of_sats": int(block.split()[1].strip()),
                "obs_types": [element.strip() for element in block.split()[2:]],
            }
        )
    return {content["constellation"]: content for content in types}


def parse(file_path):
    df = pd.read_csv(file_path, sep="|", header=None)
    df.columns = ["lines"]
    i_end_of_header = df[df.lines.str.contains("END OF HEADER")].index[0]
    header = df.iloc[:i_end_of_header]
    obs_types = get_obs_types(header)
    df = df.iloc[i_end_of_header + 1 :].reset_index(drop=True)
    df["i"] = df.index
    is_timestamp = df.lines.str.startswith(">")
    timestamps = df[is_timestamp]
    timestamps["time"] = pd.to_datetime(
        timestamps.lines.str[2:29], format="%Y %m %d %H %M %S.%f"
    )
    timestamps = timestamps.drop(columns="lines")
    df = pd.merge_asof(df, timestamps, on="i", direction="backward").drop(columns="i")[
        ~is_timestamp
    ]
    df = df[~is_timestamp]
    df.columns = ["records", "time"]
    df["records"] = df.records.str.pad(
        df.records.str.len().max(), side="right", fillchar=" "
    )
    df["sv"] = df.records.str[:3]
    df["records"] = df.records.str[3:]
    # See table A3 in the RINEX 3.05 specification
    block_length = 14 + 1 + 1
    df.records = df.records.str.wrap(block_length, drop_whitespace=False)
    obs = (
        df.records.str.split("\n", expand=True)
        .stack()
        .str[:14]
        .str.strip()
        .replace({"": "NaN"})
        .unstack()
        .astype(float)
    )
    obs["sv"] = df.sv
    obs["constellation"] = df.sv.str[0]
    obs["time"] = df.time
    group_dfs = []
    for constellation, group_df in obs.groupby("constellation"):
        group_df.set_index(["time", "constellation", "sv"], inplace=True)
        types = obs_types[constellation]["obs_types"]
        group_df = group_df.iloc[:, : len(types)]
        group_df.columns = types
        group_df = group_df.stack().reset_index(drop=False)
        group_df.columns = ["time", "constellation", "sv", "obs_type", "obs_value"]
        group_df = group_df.drop(columns=["constellation"])
        group_df = group_df[["time", "sv", "obs_value", "obs_type"]]
        group_dfs.append(group_df)
    return pd.concat(group_dfs).sort_values(by=["time"])
