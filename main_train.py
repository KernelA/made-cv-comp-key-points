from typing import Optional
import pathlib

import configargparse
import torch
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from torchvision import transforms
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


from face_landmarks.dataset.landmarks import FullLandmarkDataModule
from cli_utils import is_dir
from face_landmarks import ModelTrain, TrainTestLandmarkDataModule, TORCHVISION_RGB_MEAN, \
    TORCHVISION_RGB_STD, FullLandmarkDataModule, WingLoss
from model import LandmarkPredictor
from params import LossParams, TrainParams, RANDOM_STATE, OptimizerParams


def train_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.get_default_dtype()),
        transforms.Resize((192, 256)),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)
    ])


def valid_transform():
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.get_default_dtype()),
        transforms.Resize((192, 256)),
        transforms.Normalize(mean=TORCHVISION_RGB_MEAN, std=TORCHVISION_RGB_STD)
    ])


def get_model(num_landmarks: int, dropout_prob: float):
    pretrained_model = models.vgg13_bn(pretrained=True)
    backbone = pretrained_model.features

    return LandmarkPredictor(backbone=backbone, emb_dim=512,
                             num_landmarks=num_landmarks, dropout_prob=dropout_prob)


def get_loss(w: float, eps: float, reduction: Optional[str]):
    return torch.jit.script(WingLoss(w, eps, reduction))


def main(args):
    pl.seed_everything(RANDOM_STATE)

    exp_dir = pathlib.Path(args.exp_dir)
    exp_dir.mkdir(exist_ok=True, parents=True)

    train_params = TrainParams()

    train_tr, valid_tr = train_transform(), valid_transform()

    if train_params.train_size == 1:
        datamodule = FullLandmarkDataModule(path_to_dir=args.data_dir,
                                            train_batch_size=args.train_batch_size,
                                            val_batch_size=args.valid_batch_size,
                                            train_num_workers=args.train_num_workers,
                                            valid_num_workers=args.valid_num_workers,
                                            random_state=RANDOM_STATE,
                                            train_transforms=train_tr)
    else:
        datamodule = TrainTestLandmarkDataModule(path_to_dir=args.data_dir,
                                                 train_batch_size=args.train_batch_size,
                                                 val_batch_size=args.valid_batch_size,
                                                 train_num_workers=args.train_num_workers,
                                                 valid_num_workers=args.valid_num_workers,
                                                 random_state=RANDOM_STATE,
                                                 train_size=train_params.train_size,
                                                 train_transforms=train_tr, val_transforms=valid_tr)

    model = get_model(train_params.num_landmarks, train_params.dropout_prob)

    loss_params = LossParams()

    loss = get_loss(loss_params.w, loss_params.eps, loss_params.redcution)

    opt_params = OptimizerParams()

    train_module = ModelTrain(model, loss, opt_params)

    checkpoint_dir = exp_dir / "checkpoint"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = callbacks.ModelCheckpoint(monitor="Val MSE loss",
                                                    dirpath=checkpoint_dir,
                                                    filename="{epoch}-{Val MSE loss:.2f}",
                                                    verbose=True,
                                                    save_last=True,
                                                    save_top_k=2,
                                                    mode="min",
                                                    save_weights_only=False)

    log_dir = exp_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    logger = TensorBoardLogger(str(log_dir))

    gpus = -1 if torch.cuda.is_available() else 0

    trainer = pl.Trainer(amp_backend="native",
                         auto_scale_batch_size=True,
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

    args = parser.parse_args()

    main(args)
