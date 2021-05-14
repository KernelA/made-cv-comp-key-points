import os
from multiprocessing import Pool
import shelve
import pathlib
import shutil
import itertools
from torchvision.transforms.transforms import Resize

import tqdm
import numpy as np
import configargparse
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torchvision import transforms
from torchvision import models

from cli_utils import is_dir
from face_landmarks import LandMarkDatset
from face_landmarks.costants import TORCHVISION_RGB_MEAN, TORCHVISION_RGB_STD, PICKLE_PROTOCOl


def get_feature_extarctor(model):
    feature_extarctor = model.features
    feature_extarctor.eval()
    for param in feature_extarctor.parameters():
        param.requires_grad = False

    return feature_extarctor


@torch.no_grad()
def extract_features(train_dir: str, sample_indices, db_dir: str, part: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_transform = transforms.Compose([transforms.ConvertImageDtype(
        torch.float32), transforms.Resize(), transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)])

    feature_extarctor = get_feature_extarctor(
        models.densenet121(pretrained=True)).to(device)

    dataset = Subset(LandMarkDatset(train_dir, is_train=True,
                                    transformations=inference_transform), sample_indices)

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        pin_memory=True, drop_last=False, num_workers=0)

    with shelve.open(os.path.join(db_dir, f"features_{part}"), flag="c", protocol=PICKLE_PROTOCOl) as db:
        for single_image_data in tqdm.tqdm(loader):
            features = F.adaptive_avg_pool2d(feature_extarctor(
                single_image_data["image"].to(device)), 1).squeeze()
            db[single_image_data["image_name"][0]] = features.cpu().numpy()


def main(args):
    db_dir = pathlib.Path(args.db_dir)

    if db_dir.exists():
        shutil.rmtree(str(db_dir))

    db_dir.mkdir(parents=True)

    data = LandMarkDatset(args.data_dir, is_train=True, transformations=None)
    total_examples = len(data)

    sample_indices = tuple(range(total_examples))

    chunk_parts = args.num_workers
    chunk_size = len(sample_indices) // chunk_parts
    if chunk_size == 0:
        chunk_size = len(sample_indices)

    chunks = []

    for i in range(chunk_parts):
        start_index = i * chunk_parts
        end = start_index + chunk_size
        if i == chunk_parts - 1:
            chunks.append(sample_indices[start_index:])
        else:
            chunks.append(sample_indices[start_index: end])

    with Pool(args.num_workers) as workers:
        workers.starmap(extract_features, zip(
            itertools.repeat(args.data_dir, len(chunks)),
            chunks,
            itertools.repeat(str(db_dir), len(chunks)),
            tuple(range(1, len(chunks) + 1))
        )
        )


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", required=True, is_config_file=True, help="A config file")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--data_dir", required=True, type=is_dir)
    parser.add_argument("--db_dir", type=str, required=True)

    args = parser.parse_args()

    main(args)
