import os
import logging

import torch
from torch.utils import data
from torchvision import io
import pytorch_lightning as pl
import pandas as pd
from torchvision.io.image import ImageReadMode


class LandMarkDatset(data.Dataset):
    def __init__(self, path_to_dir: str, is_train: bool, transformations) -> None:
        assert os.path.isdir(path_to_dir)
        super().__init__()
        self._logger = logging.getLogger("kp.dataset")
        self._path_to_dir = path_to_dir
        self._is_train = is_train
        self._transformations = transformations
        self._image_dir = os.path.join(self._path_to_dir, "train" if self._is_train else "test")
        csv_landmarks_date = os.path.join(
            path_to_dir, "landmarks.csv") if is_train else "test_points.csv"
        landmarks_data = pd.read_csv(
            csv_landmarks_date, sep="\t", index="file_name", engine="c")

        bad_files = self._check_bad_images()

        if bad_files:
            self._logger.info("Bad images: %d from %d", len(bad_files), landmarks_data.shape[0])
            landmarks_data.drop(index=bad_files, inplace=True)

        self._image_names = landmarks_data.index.to_list()
        self._landmarks_points = torch.from_numpy(landmarks_data.to_numpy())

    def _check_bad_images(self, landmarks_data) -> list:
        self._logger.info("Check images for bad data")
        bad_files = []

        for image_name in landmarks_data.index:
            try:
                self._read_image(image_name)
            except Exception:
                self._logger.exception("Detected bad image")
                bad_files.append(image_name)

        return bad_files

    def _read_image(self, image_name: str):
        image_path = os.path.join(self._image_dir, image_name)
        return io.read_image(image_path, ImageReadMode.RGB)

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        image = self._read_image(self._image_names[index])
        if self._transformations is not None:
            image = self._transformations(image)

        return {"image": image, "landmarks": self._landmarks_points[index]}


# class LandmarkData(pl.LightningDataModule):
#     def __init__(self, path_to_dir: str, batch_size: int, train_train_transforms=None,
#                  val_transforms=None, test_transforms=None, dims=None):
#         assert os.path.isfile(path_to_dir), f"Input '{path_to_dir}' does not exist"
#         super().__init__(train_transforms=train_train_transforms,
#                          val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
#         self._zip_acthive = path_to_dir
#         self._batch_size = batch_size
