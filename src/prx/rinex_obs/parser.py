import pandas as pd


def get_obs_types(header):
    types = []
    marker = "SYS / # / OBS TYPES"
    for line in header.lines.to_list():
        if marker in line:
            line = line.replace(marker, "")
            blocks = [s.strip() for s in line.split()]
            if len(blocks[0]) < 3:
                constellation = blocks[0]
                number_of_sats = int(blocks[1])
                obs_types = blocks[2:]
                types.append({"constellation": constellation, "number_of_sats": number_of_sats, "obs_types": obs_types})
            else:
                # This is a continuation line:
                obs_types = line.split()
                types[-1]["obs_types"].extend(obs_types)
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
