import csv
from face_landmarks.dataset import transforms
import json
import logging
import os
import shutil
import pathlib
import random

import configargparse
import torch
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm
from matplotlib import pyplot as plt

from main_train import get_model
from face_landmarks import LandMarkDatset, ModelTrain, denormalize_tensor_to_image, read_image
from cli_utils import is_dir, is_file
from params import TrainParams
from main_train import valid_transform


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


@torch.no_grad()
def make_prediction(model, data_loader, pred_file, debug_dir):

    debug_dir_pathlib = None if debug_dir is None else pathlib.Path(debug_dir)

    if debug_dir_pathlib is not None:
        if debug_dir_pathlib.exists():
            shutil.rmtree(debug_dir_pathlib)
        debug_dir_pathlib.mkdir(parents=True)

    def save_debug_image(image, landmarks, image_name, debug_dir):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image.permute(1, 2, 0))
        ax.plot(landmarks[:, 0], landmarks[:, 1], "go")
        fig.savefig(debug_dir / image_name, bbox_inches="tight")
        plt.close(fig)

    with open(pred_file, "w", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow(
            ["file_name"] + sum([[f"Point_M{i}_X", f"Point_M{i}_Y"] for i in range(30)], []))

        debug_batch_indices = random.sample(tuple(range(len(data_loader))), k=20)

        for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            batch_size = batch["image"].shape[0]
            predicted_landmarks = model(batch).view(batch_size, -1, 2)

            fs = batch["scale_coef"]  # B
            margins_x = batch["crop_margin_x"]  # B
            margins_y = batch["crop_margin_y"]  # B

            restored_landmarks = torch.round(restore_landmarks_batch(
                torch.clone(predicted_landmarks), fs, margins_x, margins_y)).cpu().to(torch.long)  # B x NUM_PTS x 2

            if batch_idx in debug_batch_indices:
                try:
                    selected_index = random.randint(0, batch_size - 1)
                    image = read_image(batch["image_path"][selected_index])
                    save_debug_image(image, restored_landmarks[selected_index],
                                     batch["image_name"][selected_index], debug_dir_pathlib)
                except Exception:
                    logging.getLogger().exception("Unexpected error in debug save")

            for num_image, (restored_landmark_pos, indices) in enumerate(zip(restored_landmarks,
                                                                             batch["point_indices"])):
                selected_points = torch.flatten(torch.index_select(
                    restored_landmark_pos, dim=0, index=indices)).numpy()

                csv_writer.writerow([batch["image_name"][num_image]] +
                                    list(selected_points))

    logging.getLogger().info("Save predictions to '%s'", args.pred_file)


@torch.no_grad()
def error_anal(model, data_loader):
    loss_values = dict()

    for batch in tqdm(data_loader, total=len(data_loader)):
        batch_size = batch["image"].shape[0]
        predicted_landmarks = model(batch).view(batch_size, -1)

        error = torch.mean(F.mse_loss(
            predicted_landmarks, batch["landmarks"].view(batch_size, -1), reduction="none"), dim=1)

        for i, image_name in enumerate(batch["image_name"]):
            loss_values[image_name] = error[i].item()

    return loss_values


@torch.no_grad()
def main(args):
    train_params = TrainParams()

    dataset = LandMarkDatset(path_to_dir=args.data_dir, is_train=args.error_anal, annot_file=None,
                             transformations=valid_transform(train_params.img_size_in_batch))

    if args.precompute:
        dump_dir = os.path.join(args.data_dir, "dump")

        if os.path.isdir(dump_dir):
            shutil.rmtree(dump_dir)
        os.makedirs(dump_dir, exist_ok=True)
        dataset.precompute(dump_dir)

    test_loader = data.DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, pin_memory=True)

    model = get_model(train_params.num_landmarks, train_params.dropout_prob,
                      train_params.train_backbone)

    location = "cuda" if torch.cuda.is_available() else "cpu"

    model = ModelTrain.load_from_checkpoint(
        args.checkpoint, map_location=location,
        model=model, loss_func=None,
        train_backbone_after_epoch=None,
        optimizer_params=None, scheduler_params=None, target_metric_name=None)
    model.freeze()

    if args.error_anal:
        error_by_image = error_anal(model, test_loader)
        with open(args.pred_file, "w", encoding="utf-8") as file:
            json.dump(error_by_image, file)
        logging.getLogger().info("Save errors to '%s'", args.pred_file)
    else:
        make_prediction(model, test_loader, args.pred_file, args.debug_dir)


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", is_config_file=True)
    parser.add_argument("--data_dir", type=is_dir, required=True)
    parser.add_argument("--pred_file", type=str, required=True, help="A path to save predictions")
    parser.add_argument("--checkpoint", type=is_file, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--error_anal", action="store_true")
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--debug_dir", default=None, type=str,
                        help="A path to dir with predicted landmarks")

    args = parser.parse_args()

    main(args)
