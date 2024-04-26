from glob import glob
import platform
import subprocess
from pathlib import Path
import os
import shutil
import gzip
import src.prx.main as prx
import src.prx.helpers as helpers
import src.prx.constants as constants

log = helpers.get_logger(__name__)
remote_dataset_filepath = Path("C:/Users/paul/OneDrive - enac.fr/DataCollect_GoogleSDC_2021")

if __name__ == "__main__":
    """
    This script creates a local copy of gunzipped rinex files from the GSDC2021 data set and process them with prx
    
    The filepath to the original GSDC2021 folder is specified in `remote_dataset_filepath`
    """

    # discover gsdc 2021 rinex files
    remote_rinex_list = list(remote_dataset_filepath.glob("**/*.20o")) + list(
        remote_dataset_filepath.glob("**/*.21o")
    )

    print(f"{len(remote_rinex_list)} RINEX files discovered")

    # iterate over rinex files
    local_dataset_directory = Path(f"./GSDC2021").resolve()
    local_gz_list = []
    for file in remote_rinex_list:
        print(file)
        relative_path = os.path.relpath(Path(f"{str(file)}.gz"), remote_dataset_filepath)
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

    # process single file with prx
    prx.process(
        observation_file_path=local_gz_list[0],
        output_format="csv",
    )
