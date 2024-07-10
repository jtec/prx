import pandas as pd
import re


def get_obs_types(header):
    types = []
    marker = "SYS / # / OBS TYPES"
    lines = " " + " ".join(header[header.lines.str.contains(marker)].lines.to_list()).replace(marker, " ")
    blocks = re.split(r'( C | E | J | R | I | G| S )', lines)
    blocks = [block.strip() for block in blocks if len(block.strip()) > 0]
    blocks = [block + " " + blocks[i + 1] for i, block in enumerate(blocks) if len(block) == 1]
    for block in blocks:
        types.append({"constellation": block.split()[0].strip(), "number_of_sats": int(block.split()[1].strip()),
                      "obs_types": [element.strip() for element in block.split()[2:-1]]})
    return types


def parse(file_path):
    df = pd.read_csv(file_path, sep='|', header=None)
    df.columns = ["lines"]
    i_end_of_header = df[df.lines.str.contains('END OF HEADER')].index[0]
    header = df.iloc[:i_end_of_header]
    obs_types = get_obs_types(header)
    df = df.iloc[i_end_of_header + 1:].reset_index(drop=True)
    df["i"] = df.index
    is_timestamp = df.lines.str.startswith('>')
    timestamps = df[df.lines.str.startswith('>')]
    timestamps['rx_time'] = pd.to_datetime(timestamps.lines.str[2:29], format="%Y %m %d %H %M %S.%f")
    timestamps = timestamps.drop(columns='lines')
    df = pd.merge_asof(df, timestamps, on='i', direction='backward').drop(columns='i')[~is_timestamp]
    df = df[~is_timestamp]
    df.columns = ["records", "timestamps"]
    df["sv"] = df.records.str[:3]
    ...
