import csv
import logging

import configargparse
import torch
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
def main(args):
    dataset = LandMarkDatset(path_to_dir=args.data_dir, is_train=False,
                             transformations=get_transforms())

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

    with open(args.pred_file, "w", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(
            ["file_name"] + sum([[f"Point_M{i}_X", f"Point_M{i}_Y"] for i in range(30)], start=[]))

        for batch in tqdm(test_loader, total=len(test_loader)):
            batch_size = batch["image"].shape[0]
            predicted_landmarks = model(batch).view(batch_size, -1, 2)

            for num_image, (normalized_landmark_pos, indices) in enumerate(zip(predicted_landmarks, batch["point_indices"])):
                height, width = batch["img_height"][num_image].item(
                ), batch["img_width"][num_image].item()
                landmark_pos = denormalize_landmarks(normalized_landmark_pos, width, height)
                selected_points = torch.index_select(landmark_pos, dim=0, index=indices).cpu()
                csv_writer.writerow(selected_points)

    logging.getLogger().info("Save predcition to '%s'", args.pred_file)


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", is_config_file=True)
    parser.add_argument("--data_dir", type=is_dir, required=True)
    parser.add_argument("--pred_file", type=str, required=True, help="A path to save predictions")
    parser.add_argument("--checkpoint", type=is_file, required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    main(args)
