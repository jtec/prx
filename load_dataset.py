import os
from time import time
from pathlib import Path
import json
import logging
import pandas as pandas
from cachetools import cached
import functools
from multiprocessing.connection import Client

logger = logging.getLogger(f"phase {__name__}")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

cache_coms_timeout_s = 1e3

def load_dataset(dataset_name):
    print(f"Loading dataset {dataset_name}")
    cache_key = __name__ + dataset_name
    t0 = time()
    cached_ds = get_cached(cache_key)
    if cached_ds is not None:
        print(f"Loading dataset from cache took [s] {time() - t0:.6f}")
        return cached_ds
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
    set_cached(cache_key, ds)
    return ds

def get_cached(key):
    answer = None
    connection_to_pycache = Client(('localhost', 6000),
                                   authkey=bytes("pw", encoding='utf-8'))
    connection_to_pycache.send({"key": key})
    t_send = time()
    while time() - t_send < cache_coms_timeout_s:
        if connection_to_pycache.poll():
            answer = connection_to_pycache.recv()
            break
    connection_to_pycache.send("close")
    connection_to_pycache.close()
    return answer


def set_cached(key, data):
    connection_to_pycache = Client(('localhost', 6000),
                                   authkey=bytes("pw", encoding='utf-8'))
    connection_to_pycache.send({"key": key, "data": data})
    t_send = time()
    while time() - t_send < cache_coms_timeout_s:
        if connection_to_pycache.poll():
            answer = connection_to_pycache.recv()
            break
    connection_to_pycache.send("close")
    connection_to_pycache.close()


def dataset2dictionary(dataset_root):
    t0 = time()
    file_parsers = [parse_json, parse_csv]
    # Build dictionary representation of the dataset:
    def parse_directory(directory_path, dataset_root_dir=None):
        # Catch when this function is called for the data set root, which we need to build relative paths in the
        # following recursive calls:
        if dataset_root_dir is None:
            dataset_root_dir = directory_path
        directory_node = {}
        for subdirectory_or_file in directory_path.iterdir():
            if subdirectory_or_file.is_file():
                # If we see a file, we just parse it or store its path so we can e.g. parse it at a later stage.
                directory_node[subdirectory_or_file.name] = subdirectory_or_file.relative_to(dataset_root_dir)
                for parser in file_parsers:
                    parsed = parser(subdirectory_or_file)
                    if parsed is not None:
                        directory_node[subdirectory_or_file.name] = parsed
            if subdirectory_or_file.is_dir():
                # For a directory, we create a new node and treat it just like the root node:
                directory_node[subdirectory_or_file.name] = {}
                directory_node[subdirectory_or_file.name] = parse_directory(subdirectory_or_file,
                                                                            dataset_root_dir=dataset_root_dir)
        return directory_node

    ds = parse_directory(dataset_root)
    logger.info(f"Parsing dataset files took [s] {time() - t0:.6f}")
    return ds


def parse_json(absolute_file_path):
    if ".json" not in str(absolute_file_path)[-5:]:
        return None
    try:
        with open(absolute_file_path) as file:
            data = json.load(file)
        return data
    except Exception as e:
        logger.warning("Exception trying to parse JSON file: "+ str(e))
        return None


def parse_csv(absolute_file_path):
    if ".csv" not in str(absolute_file_path)[-4:]:
        return None
    try:
        # Count number of comment lines:
        comments = []
        first_few_lines = []
        with open(absolute_file_path) as file:
            for line in file.readlines():
                if "#" in line[0:1]:
                    comments.append(line)
                else:
                    if len(first_few_lines) < 10:
                        first_few_lines.append(line)
        # Detect files that have only comments and no data:
        if len(first_few_lines) < 3:
            return None

        with open(absolute_file_path) as file:
            data = pandas.read_csv(file, header=len(comments))

        # Merge units into column names:
        new_column_names = {}
        for column_name in data.columns:
            new_column_names[column_name] = f"{column_name} [{data[column_name][0]}]"
        data.rename(columns=new_column_names, inplace=True)
        data.drop(0, inplace=True)
        return data
    except Exception as e:
        logger.warning("Exception trying to parse CSV file: " + str(e))
        return None

