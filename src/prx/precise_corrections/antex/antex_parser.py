from pathlib import Path
import pandas as pd
import numpy as np

from prx.util import ecef_2_satellite

# def parquet_loading(func):
#     """
#     This is a decorator to check if a .parquet file exists and loads it,
#     rather than parsing the original file.

#     NOTE: the parquet files must be manually erased when a modification
#     is introduced in the parser.

#     To use it, use the @parquet_loading decorator before the parsing
#     function definition.

#     Example:
#         @parquet_loading
#         def prx_to_pandas(path_csv: Path, observation_filter: dict = {}):
#         ...
#     """

#     def wrapper(path, **kwargs):
#         if path.with_suffix(".parquet").exists():
#             print(f"Reading '{path.with_suffix('.parquet')}'")
#             df = pd.read_parquet(path.with_suffix(".parquet"))
#         else:
#             print(f"Reading '{path.resolve()}' file and saving it as .parquet")
#             df = func(path, **kwargs)
#             df.to_parquet(path.with_suffix(".parquet"))
#         return df

#     return wrapper


# @parquet_loading
def parse_atx(filepath: Path):
    """
    ANTEX file format description is available at https://files.igs.org/pub/data/format/antex14.txt
    """

    def read_antenna(file):
        atx_df = pd.DataFrame(
            columns=[
                "antenna_type",
                "satellite_or_serial_no",
                "valid_from",
                "valid_until",
                "constellation",
                "carrier_freq_id",
                "pco_north_m",
                "pco_east_m",
                "pco_up_m",
            ]
        )
        # line with TYPE / SERIAL NO
        line = file.readline()
        antenna_type = line[0:20].strip()
        satellite = line[20:40].strip()

        # line with # of FREQUENCIES
        for _ in range(4):
            line = file.readline()
        nb_freq = int(line[0:6])

        # line with VALID FROM
        line = file.readline()
        if "VALID FROM" in line:
            valid_from = pd.Timestamp(
                year=int(line[0:6]),
                month=int(line[6:12]),
                day=int(line[12:18]),
                hour=int(line[18:24]),
                minute=int(line[24:30]),
                second=int(line[30:35]),
                microsecond=int(np.floor(float(line[36:42]))),
            )
            line = file.readline()
        else:
            valid_from = pd.NaT

        # line with VALID UNTIL
        if "VALID UNTIL" in line:
            valid_until = pd.Timestamp(
                year=int(line[0:6]),
                month=int(line[6:12]),
                day=int(line[12:18]),
                hour=int(line[18:24]),
                minute=int(line[24:30]),
                second=int(line[30:35]),
                microsecond=int(np.floor(float(line[36:42]))),
            )
        else:
            valid_until = pd.NaT

        # skip line with SINEX CODE or COMMENT
        while "START OF FREQUENCY" not in line:
            line = file.readline()

        for index in range(nb_freq):
            # line with START OF FREQUENCY
            constellation = line[3]
            carrier_freq_id = int(line[4:6])

            # line with NORTH / EAST / UP
            line = file.readline()
            north = float(line[0:10]) * 1e-3
            east = float(line[10:20]) * 1e-3
            up = float(line[20:30]) * 1e-3

            atx_df.loc[index] = pd.Series(
                {
                    "antenna_type": antenna_type,
                    "satellite_or_serial_no": satellite,
                    "valid_from": valid_from,
                    "valid_until": valid_until,
                    "constellation": constellation,
                    "carrier_freq_id": carrier_freq_id,
                    "pco_north_m": north,
                    "pco_east_m": east,
                    "pco_up_m": up,
                }
            )

            # skip lines until line with END OF FREQUENCY
            while "END OF FREQUENCY" not in line:
                line = file.readline()
            line = file.readline()

        return file, atx_df

    with open(filepath) as file:
        atx_df = pd.DataFrame()
        while True:
            line = file.readline()
            if not line:  # check if the end of file has been reached
                break
            if "START OF ANTENNA" in line:
                file, atx_df_ant = read_antenna(file)
                atx_df = pd.concat([atx_df, atx_df_ant], ignore_index=True)

    return atx_df


def compute_pco_sat(
    sat_id: np.array, sat_pos: np.array, rx_pos: np.array, epochs: np.array, atx_df
) -> np.array:
    """
    Compute the satellite Phase Center Offset

    Reference: RTKLIB manual v2.4.2, p 173
    """
    query = pd.DataFrame(
        {
            "satellite": sat_id,
            "epoch": epochs,
        }
    )

    pco_sat = pd.DataFrame(
        columns=[
            "satellite_or_serial_no",
            "carrier_freq_id",
            "pco_north_m",
            "pco_east_m",
            "pco_up_m",
        ]
    )
    for sat, group in query.groupby("satellite"):
        epoch_min = group.epoch.min()
        epoch_max = group.epoch.max()
        to_be_apended = atx_df.loc[
            (atx_df.satellite_or_serial_no == sat)
            & (atx_df.valid_from < epoch_min)
            & ((atx_df.valid_until > epoch_max) | (atx_df.valid_until.isnull())),
            [
                "satellite_or_serial_no",
                "carrier_freq_id",
                "pco_north_m",
                "pco_east_m",
                "pco_up_m",
            ],
        ]
        if len(pco_sat) == 0:
            pco_sat = to_be_apended
        else:
            pco_sat = pd.concat([pco_sat, to_be_apended], ignore_index=True)
    # rename columns
    pco_sat = pco_sat.rename(
        columns={
            "satellite_or_serial_no": "satellite",
            "pco_north_m": "pco_x_m",
            "pco_east_m": "pco_y_m",
            "pco_up_m": "pco_z_m",
        }
    )

    freq_list = pco_sat.carrier_freq_id.unique()

    # unstack carrier_freq_id
    pco_sat = pco_sat.set_index(["satellite", "carrier_freq_id"]).unstack(1)
    # rename columns
    pco_sat.columns = [
        cols[0][0:4] + str(cols[1]) + cols[0][3:] for cols in pco_sat.columns
    ]
    pco_sat = pco_sat.reindex(sorted(pco_sat.columns), axis=1)

    _, rot_mat_ecef2sat = ecef_2_satellite(sat_pos, sat_pos, epochs)

    pco_effect_list = []
    for freq in freq_list:
        pco_ecef = np.array(
            [
                rot_mat_ecef2sat[idx, :, :].T
                @ pco_sat.loc[
                    sat,
                    [
                        "pco_" + str(freq) + "_x_m",
                        "pco_" + str(freq) + "_y_m",
                        "pco_" + str(freq) + "_z_m",
                    ],
                ]
                .to_numpy()
                .reshape(-1, 1)
                for idx, sat in enumerate(sat_id)
            ]
        ).reshape(len(sat_id), 3)

        pco_effect_list.append(
            np.vecdot(
                pco_ecef,
                (sat_pos - rx_pos)
                / np.linalg.norm(sat_pos - rx_pos, axis=1).reshape(-1, 1),
            )
        )

    pco_effect = np.stack(pco_effect_list, axis=1)
    return pco_effect, freq_list

