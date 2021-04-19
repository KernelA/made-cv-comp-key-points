import csv
import json
import logging
import os
import shutil

import configargparse
import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from main_train import get_model
from face_landmarks import LandMarkDatset, TORCHVISION_RGB_MEAN, TORCHVISION_RGB_STD, ModelTrain, denormalize_landmarks
from cli_utils import is_dir, is_file
from params import TrainParams


def get_transforms():
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.get_default_dtype()),
            transforms.Resize((192, 256)),
            transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)
        ]
    )


@torch.no_grad()
def make_prediction(model, data_loader, pred_file):
    with open(pred_file, "w", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(
            ["file_name"] + sum([[f"Point_M{i}_X", f"Point_M{i}_Y"] for i in range(30)], start=[]))

        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_size = batch["image"].shape[0]
            predicted_landmarks = model(batch).view(batch_size, -1, 2)

            for num_image, (normalized_landmark_pos, indices) in enumerate(zip(predicted_landmarks, batch["point_indices"])):
                height, width = batch["img_height"][num_image].item(
                ), batch["img_width"][num_image].item()
                landmark_pos = denormalize_landmarks(normalized_landmark_pos, width, height)
                selected_points = torch.flatten(torch.index_select(
                    landmark_pos, dim=0, index=indices).cpu().to(torch.long))
                csv_writer.writerow([batch["image_name"][num_image]] +
                                    list(selected_points))

    logging.getLogger().info("Save predcition to '%s'", args.pred_file)


@torch.no_grad()
def error_anal(model, data_loader):
    loss_values = dict()

    for batch in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch["image"].shape[0]
        predicted_landmarks = model(batch).view(batch_size, -1)

        error = torch.mean(F.mse_loss(
            predicted_landmarks, batch["norm_landmarks"].view(batch_size, -1), reduction="none"), dim=1)

        for i, image_name in enumerate(batch["image_name"]):
            loss_values[image_name] = error[i].item()

    return loss_values


@torch.no_grad()
def main(args):
    dataset = LandMarkDatset(path_to_dir=args.data_dir, is_train=args.error_anal,
                             transformations=get_transforms())

    dump_dir = os.path.join(args.data_dir, "dump")

    if os.path.isdir(dump_dir):
        shutil.rmtree(dump_dir)
    os.makedirs(dump_dir, exist_ok=True)
    dataset.precompute(dump_dir)

    test_loader = data.DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, pin_memory=True)

    train_params = TrainParams()

    model = get_model(train_params.num_landmarks, train_params.dropout_prob,
                      train_params.train_backbone)

    location = "cuda" if torch.cuda.is_available() else "cpu"

    model = ModelTrain.load_from_checkpoint(
        args.checkpoint, map_location=location,
        model=model, loss_func=None,
        optimizer_params=None, target_metric_name=None)
    model.freeze()

    if args.error_anal:
        error_by_image = error_anal(model, test_loader)
        with open(args.pred_file, "w", encoding="utf-8") as file:
            json.dump(error_by_image, file)
        logging.getLogger().info("Save errors to '%s'", args.pred_file)
    else:
        make_prediction(model, test_loader, args.pred_file)


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", is_config_file=True)
    parser.add_argument("--data_dir", type=is_dir, required=True)
    parser.add_argument("--pred_file", type=str, required=True, help="A path to save predictions")
    parser.add_argument("--checkpoint", type=is_file, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--error_anal", action="store_true")

    args = parser.parse_args()

    main(args)
