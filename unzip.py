import zipfile
import pathlib

import configargparse

from cli_utils import is_file


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Extract data to: ", out_dir)

    with zipfile.ZipFile(args.zip, "r") as zip_file:
        zip_file.extractall(out_dir)


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", required=True, is_config_file=True, help="A config file")
    parser.add_argument("--zip", type=is_file, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)
