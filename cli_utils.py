import os

import configargparse


def is_file(arg: str) -> str:
    if not os.path.isfile(arg):
        raise configargparse.ArgumentTypeError(f"'{arg}' is not a file")

    return arg


def is_dir(arg: str) -> str:
    if not os.path.isdir(arg):
        raise configargparse.ArgumentTypeError(f"'{arg}' is not a directory")

    return arg
