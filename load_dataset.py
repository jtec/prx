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
    def parse_directory(directory_path, dataset_root_dir=""):
        directory_node = {}
        for subdirectory_or_file in directory_path.iterdir():
            if subdirectory_or_file.is_file():
                # If we see a file, we just parse it or store its path so we can e.g. parse it at a later stage.
                directory_node[subdirectory_or_file.name] = subdirectory_or_file.relative_to(dataset_root)
            if subdirectory_or_file.is_dir():
                # For a directory, we create a new node and treat it just like the root node:
                directory_node[subdirectory_or_file.name] = {}
                directory_node[subdirectory_or_file.name] = parse_directory(subdirectory_or_file,
                                                                            dataset_root_dir=dataset_root)
        return directory_node

    ds = parse_directory(dataset_root, dataset_root_dir=dataset_root)
    return ds