import logging
from typing import Optional, Tuple
import pathlib

import configargparse
import torch
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torchvision import transforms
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


from cli_utils import is_dir, is_file
from face_landmarks import ModelTrain, TrainTestLandmarkDataModule, TORCHVISION_RGB_MEAN, \
    TORCHVISION_RGB_STD, FullLandmarkDataModule, WingLoss
from model import LandmarkPredictor
from params import LossParams, SchedulerPrams, TrainParams, RANDOM_STATE, OptimizerParams


def train_transform(img_size: Tuple[int, int]):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.get_default_dtype()),
        transforms.Resize(img_size),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)
    ])


def valid_transform(img_size: Tuple[int, int]):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.get_default_dtype()),
        transforms.Resize(img_size),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)
    ])


def get_model(num_landmarks: int, dropout_prob: float, train_backbone: bool):
    backbone = models.resnet18(pretrained=True)

    return LandmarkPredictor(backbone=backbone, emb_dim=512,
                             num_landmarks=num_landmarks, dropout_prob=dropout_prob, train_backbone=train_backbone)


def get_loss(w: float, eps: float, reduction: Optional[str]):
    return torch.jit.script(WingLoss(w, eps, reduction))


def load_ignore_images(path_to_images):
    with open(path_to_images, "r", encoding="utf-8") as file:
        return list(map(str.strip, file.readlines()))


def main(args):
    pl.seed_everything(RANDOM_STATE)

    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    train_params = TrainParams()

    train_tr, valid_tr = train_transform(
        train_params.img_size_in_batch), valid_transform(train_params.img_size_in_batch)

    ignore_train_images_list = None

    if args.ignore_images is not None:
        ignore_train_images_list = load_ignore_images(args.ignore_images)

    if train_params.train_size == 1:
        datamodule = FullLandmarkDataModule(path_to_dir=args.data_dir,
                                            ignore_train_images=ignore_train_images_list,
                                            train_batch_size=args.train_batch_size,
                                            val_batch_size=args.valid_batch_size,
                                            train_num_workers=args.train_num_workers,
                                            valid_num_workers=args.valid_num_workers,
                                            random_state=RANDOM_STATE,
                                            train_transforms=train_tr,
                                            precompute_data=args.precompute_data)
    else:
        datamodule = TrainTestLandmarkDataModule(path_to_dir=args.data_dir,
                                                 ignore_train_images=ignore_train_images_list,
                                                 train_batch_size=args.train_batch_size,
                                                 val_batch_size=args.valid_batch_size,
                                                 train_num_workers=args.train_num_workers,
                                                 valid_num_workers=args.valid_num_workers,
                                                 random_state=RANDOM_STATE,
                                                 train_size=train_params.train_size,
                                                 precompute_data=args.precompute_data,
                                                 train_transforms=train_tr,
                                                 val_transforms=valid_tr)

    model = get_model(train_params.num_landmarks, train_params.dropout_prob,
                      train_params.train_backbone)

    loss_params = LossParams()

    loss = get_loss(loss_params.w, loss_params.eps, loss_params.redcution)

    opt_params = OptimizerParams()
    scheduler_params = SchedulerPrams()

    target_metric_name = "Val MSE loss"

    train_module = ModelTrain(model=model, loss_func=loss, optimizer_params=opt_params,
                              scheduler_params=scheduler_params,
                              target_metric_name=target_metric_name,
                              save_img_every_train_batch=25)

    checkpoint_dir = exp_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = callbacks.ModelCheckpoint(monitor=target_metric_name,
                                                    dirpath=checkpoint_dir,
                                                    filename="{epoch}-{Val MSE loss:.4f}",
                                                    verbose=True,
                                                    save_last=True,
                                                    save_top_k=2,
                                                    mode="min",
                                                    save_weights_only=False)

    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = TensorBoardLogger(str(log_dir))

    gpus = -1 if torch.cuda.is_available() else None

    if gpus is None:
        logging.getLogger().warning("GPU is not available. Try train on CPU. It may will bew very slow")

    trainer = pl.Trainer(amp_backend="native",
                         auto_scale_batch_size="binsearch",
                         gpus=gpus,
                         logger=logger,
                         auto_select_gpus=True,
                         benchmark=True,
                         check_val_every_n_epoch=train_params.check_val_every_n_epoch,
                         flush_logs_every_n_steps=train_params.flush_logs_every_n_steps,
                         default_root_dir=str(exp_dir),
                         deterministic=False,
                         fast_dev_run=args.fast_dev_run,
                         progress_bar_refresh_rate=10,
                         precision=train_params.precision,
                         max_epochs=train_params.max_epochs,
                         callbacks=[checkpoint_callback]
                         )

    trainer.fit(train_module, datamodule=datamodule)


if __name__ == '__main__':
    parser = configargparse.ArgParser()
    parser.add_argument("-c", is_config_file=True, help="A config file")
    parser.add_argument("--train_batch_size", type=int, default=128, help="A train batch size")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="A train batch size")
    parser.add_argument("--valid_num_workers", type=int, default=2,
                        help="A num workers to load valid data")
    parser.add_argument("--train_num_workers", type=int, default=2,
                        help="A num workers to load valid data")
    parser.add_argument("--data_dir", type=is_dir, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--precompute_data", action="store_true")
    parser.add_argument("--ignore_images", required=False, default=None, type=is_file,
                        help="A path to txt files with image to ignore during training (bad images)")

    args = parser.parse_args()

    main(args)
