import zipfile
import pathlib
import os
import traceback

import configargparse
from torchvision import io
from torchvision.io.image import ImageReadMode
from tqdm import tqdm
import pandas as pd

from cli_utils import is_file


def check_images(out_dir):
    for split_type, lanmark_file in zip(("train",), ("landmarks.csv",)):
        image_dir = pathlib.Path(out_dir, "contest01_data", split_type, "images")
        landmark_filepath = image_dir.parent / lanmark_file
        landmarks_data = pd.read_csv(
            landmark_filepath, sep="\t", index_col="file_name", engine="c")

        bad_images = []

        for image_name in tqdm(landmarks_data.index, desc="Check bad images"):
            try:
                io.read_image(os.path.join(image_dir, image_name), ImageReadMode.RGB)
            except Exception:
                traceback.print_exc()
                bad_images.append(image_name)

        if bad_images:
            print("Bad images: %d from %d", len(bad_images), landmarks_data.shape[0])
            landmarks_data.drop(index=bad_images, inplace=True)
            landmarks_data.to_csv(landmark_filepath, sep="\t", index=True)


def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Extract data to: ", out_dir)

    with zipfile.ZipFile(args.zip, "r") as zip_file:
        zip_file.extractall(out_dir)

    if args.check_images:
        check_images(out_dir)


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", required=False, is_config_file=True, help="A config file")
    parser.add_argument("--zip", type=is_file, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--check_images", action="store_true")

    args = parser.parse_args()

    main(args)
