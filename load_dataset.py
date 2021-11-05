import os
from pathlib import Path
import json


def load_dataset(dataset_name):
    print(f"Loading dataset {dataset_name}")
    # Do we know in which directory to find those datasets?
    this_file = Path(__file__)
    path_file = Path(this_file.parent, this_file.name.replace(".py", "_paths.json"))
    if path_file.is_file():
        with open(path_file) as file:
            parsed_path_file = json.load(file)
    else:
        static_files_directory = input("Did not find JSON file with static files directory, please enter: ")
        path_file_content = {"static_files_directory": str(Path(static_files_directory))}
        with open(path_file, 'w') as outfile:
            json.dump(path_file_content, outfile, indent=2)
        with open(path_file) as file:
            parsed_path_file = json.load(file)
    dataset_root = Path(parsed_path_file["static_files_directory"], "dataSets", dataset_name)
    ds = dataset2dictionary(dataset_root)


def dataset2dictionary(dataset_root):
    # Build dictionary representation of the dataset:
    ds = {}
    return ds
    def parse_directory(dir):
        for (root, subdirs, files) in os.walk(dir, topdown=True):
            for file in files:
                print(file)
            for subdir in subdirs:
                parse_directory(subdir)
    parse_directory(dataset_root)

    for path in dataset_root.rglob('*.json'):
        with open(path) as file:
            parsed_path_file = json.load(file)
            relative_path = str(path.relative_to(dataset_root)).split(os.path.sep)
            for level in relative_path:
                ds[level] = {}
            if len(level) == 1:
                ds[file.name] = parsed_path_file
            else:
                ds[level][file.name] = parsed_path_file
    return ds