import sys
from load_dataset import load_dataset

def main() -> int:
    dataset_name = "enac23april2021"

    ds = load_dataset(dataset_name)
    return 0


if __name__ == '__main__':
    sys.exit(main())