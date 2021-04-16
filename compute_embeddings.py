import pathlib
import pickle

import torch
import numpy as np
from torch.utils import data
import configargparse
from torchvision import transforms, models
from tqdm import tqdm

from cli_utils import is_dir
from face_landmarks import LandMarkDatset, SquarePaddingResize
from face_landmarks.costants import PICKLE_PROTOCOl, TORCHVISION_RGB_MEAN, TORCHVISION_RGB_STD


def init_model(device):
    dense_net = models.densenet121(pretrained=True)
    feature_extarctor = dense_net.features
    feature_extarctor.eval()
    feature_extarctor.to(device)
    for param in feature_extarctor.parameters():
        param.requires_grad = False

    return feature_extarctor


@torch.no_grad()
def get_features(data_loader, feature_extarctor, device):
    image_features = []

    for single_image in tqdm(data_loader):
        features = torch.nn.functional.adaptive_avg_pool2d(
            feature_extarctor(single_image["image"].to(device)), 1).squeeze()
        image_features.append(features.cpu().numpy())
    return image_features


def main(args):
    emb_path = pathlib.Path(args.emb_out)
    emb_path.parent.mkdir(exist_ok=True, parents=True)

    inference_transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        SquarePaddingResize(256),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)
    ])

    dataset = LandMarkDatset(args.data_dir, is_train=True, transformations=inference_transform)

    loader = data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = init_model(device)

    features = get_features(loader, model, device)
    features = np.concatenate(features, axis=0)

    with open(emb_path, "wb") as dump_file:
        pickle.dump(features, dump_file, protocol=PICKLE_PROTOCOl)


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add_argument("-c", is_config_file=True, help="A path to config")
    parser.add_argument("--data_dir", required=True, type=is_dir,
                        help="A base path to data dir with train test images")
    parser.add_argument("--emb_out", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="A number of workers to load data")

    args = parser.parse_args()

    main(args)
