from pathlib import Path
import os
import shutil
import gzip
import argparse
import sys
from prx.util import timeit
from prx import helpers, main

log = helpers.get_logger(__name__)


@timeit
def process_gsdc2021_dataset(path_dataset: Path):
    """
    This script creates a local copy of gunzipped RINEX files from the GSDC2021 data set and process them with prx

    The filepath to the original GSDC2021 folder is specified in `path_dataset`.
    This has to be downloaded from the Kaggle website (requires login):
    https://www.kaggle.com/competitions/google-smartphone-decimeter-challenge
    """
    # discover GSDC2021 RINEX files
    remote_rinex_list = list(path_dataset.glob("**/*.20o")) + list(
        path_dataset.glob("**/*.21o")
    )

    print(f"{len(remote_rinex_list)} RINEX files discovered")

    # iterate over RINEX files
    local_dataset_directory = Path("../../GSDC2021").resolve()
    local_gz_list = []
    for file in remote_rinex_list:
        print(file)
        relative_path = os.path.relpath(Path(f"{str(file)}.gz"), path_dataset)
        new_path = local_dataset_directory.joinpath(relative_path)
        if not new_path.exists():
            # gunzip rnx file
            with open(file, "rb") as f_in:
                with gzip.open(f"{str(file)}.gz", "wb") as file_gz:
                    shutil.copyfileobj(f_in, file_gz)

            # copy file to local dataset folder
            os.makedirs(new_path.parent, exist_ok=True)
            shutil.move(Path(file_gz.name), new_path)
        local_gz_list.append(new_path)

    assert len(local_gz_list) == 121, (
        "Something wrong went with the local copy of the GSDC2021 database"
    )

    for i, file in enumerate(local_gz_list[:]):
        main.process(
            observation_file_path=file,
            output_format="csv",
        )

    # # Parallelized version (but we should rather try to parallelize prx.process)
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=-1,verbose=100)(
    #     delayed(prx.process)(
    #         observation_file_path=file,
    #         output_format="csv",
    #     )
    #     for file in local_gz_list
    # )


if __name__ == "__main__":
    """Run this script with the following command lines in a terminal:
    > poetry env use 3.12
    > poetry install
    > poetry shell
    > python src\demos\main_gsdc2021.py --gsdc2021_folder_path "<PATH_TO_GSDC2021_FOLDER>"
    """

    parser = argparse.ArgumentParser(
        prog="prx",
        description="prx processes RINEX observations, computes a few useful things such as satellite position, "
        "relativistic effects etc. and outputs everything to a text file in a convenient format.",
        epilog="P.S. GNSS rules!",
    )
    parser.add_argument(
        "--gsdc2021_folder_path",
        type=str,
        help="GSDC2021 folder path after download from Kaggle",
        default=None,
    )
    args = parser.parse_args()
    if args.gsdc2021_folder_path is None:
        log.error("GSDC2021 dataset folder does not exist.")
        sys.exit(1)

    process_gsdc2021_dataset(Path(args.gsdc2021_folder_path))

    # # To launch this script manually, use the following lines
    # remote_dataset_filepath = Path("C:/Users/paul/backup_hard_drive/DataCollect_GoogleSDC_2021")
    # process_gsdc2021_dataset(remote_dataset_filepath)
