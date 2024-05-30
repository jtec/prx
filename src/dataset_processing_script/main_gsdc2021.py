from pathlib import Path
import os
import shutil
import gzip
import src.prx.main as prx
import src.prx.helpers as helpers


@helpers.timeit
def process_gsdc2021_dataset(path_dataset: Path):
    """
    This script creates a local copy of gunzipped RINEX files from the GSDC2021 data set and process them with prx

    The filepath to the original GSDC2021 folder is specified in `path_dataset`
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

    assert (
        len(local_gz_list) == 121
    ), "Something wrong went with the local copy of the GSDC2021 database"

    for i, file in enumerate(local_gz_list):
        prx.process(
            observation_file_path=file,
            output_format="csv",
        )

    # # Parallelized version (but we should rather try to parallelize prx.process)
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=1,verbose=100)(
    #     delayed(prx.process)(
    #         observation_file_path=file,
    #         output_format="csv",
    #     )
    #     for file in local_gz_list
    # )


if __name__ == "__main__":
    remote_dataset_filepath = Path(
        "C:/Users/paul/backup_hard_drive/DataCollect_GoogleSDC_2021"
    )
    assert remote_dataset_filepath.exists(), "GSDC2021 dataset folder does not exist"
    process_gsdc2021_dataset(remote_dataset_filepath)
