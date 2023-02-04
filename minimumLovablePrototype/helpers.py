import hashlib
from pathlib import Path


# From https://stackoverflow.com/a/3431838
def md5(file: Path):
    hash_md5 = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
