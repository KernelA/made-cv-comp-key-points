import os
import logging
from typing import Iterable, List, Union
import logging


import torch
from torch.utils import data
import pytorch_lightning as pl
import pandas as pd


from .utils import read_image, normalize_landmarks


class LandMarkDatset(data.Dataset):
    def __init__(self, *, path_to_dir: str, is_train: bool, transformations) -> None:
        assert os.path.isdir(path_to_dir)
        super().__init__()
        self._logger = logging.getLogger("kp.dataset")
        self._path_to_dir = path_to_dir
        self._is_train = is_train
        self.transformations = transformations
        split_type = "train" if self._is_train else "test"
        self._image_dir = os.path.join(
            self._path_to_dir, split_type, "images")
        csv_landmarks_date = os.path.join(
            self._path_to_dir, split_type,
            "landmarks.csv" if is_train else "test_points.csv")

        landmarks_data = pd.read_csv(
            csv_landmarks_date, sep="\t", index_col="file_name", engine="c")

        self._image_names = landmarks_data.index.tolist()
        self._landmarks_points = torch.from_numpy(landmarks_data.to_numpy()).reshape(
            len(self._image_names), len(landmarks_data.columns) // 2, 2)

    def index2img_name(self, index: Union[int, Iterable[int]]) -> Union[str, List[str]]:
        if isinstance(index, int):
            return self._image_names[index]
        else:
            return [self._image_names[i] for i in index]

    def __len__(self):
        return len(self._image_names)

    def __getitem__(self, index):
        image = read_image(os.path.join(self._image_dir, self._image_names[index]))

        height, width = image.shape[-2:]

        if self.transformations is not None:
            image = self.transformations(image)

        return {"image": image,
                "norm_landmarks": normalize_landmarks(self._landmarks_points[index], width, height),
                "image_name": self._image_names[index]}


class FullLandmarkDataModule(pl.LightningDataModule):
    def __init__(self, *, path_to_dir: str, train_batch_size: int,
                 train_num_workers: int, val_batch_size: int, valid_num_workers: int, random_state: int,
                 train_size: float = 1, train_transforms=None,
                 val_transforms=None, test_transforms=None, dims=None):
        assert os.path.isdir(path_to_dir), f"Input '{path_to_dir}' does not exist"
        assert train_batch_size > 0
        assert val_batch_size > 0
        assert train_num_workers >= 0
        assert valid_num_workers >= 0
        assert 0 < train_size <= 1
        super().__init__(train_transforms=train_transforms,
                         val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        self._data_dir = path_to_dir
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._train_size = train_size
        self._random_state = random_state
        self._train_num_workers = train_num_workers
        self._valid_num_workers = valid_num_workers
        self._train_dataset = self._test_datset = None
        self._logger = logging.getLogger("kp.datamodule")

    def setup(self, stage=None):
        general_dataset = LandMarkDatset(
            path_to_dir=self._data_dir, is_train=True, transformations=self.train_transforms)

        self._train_dataset = general_dataset

    def train_dataloader(self):
        return data.DataLoader(self._train_dataset, shuffle=True, drop_last=True,
                               batch_size=self._train_batch_size,
                               num_workers=self._train_num_workers, pin_memory=True)


class TrainTestLandmarkDataModule(FullLandmarkDataModule):
    def __init__(self, *, path_to_dir: str, train_batch_size: int,
                 train_num_workers: int, val_batch_size: int, valid_num_workers: int,
                 random_state: int, train_size: float,
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(path_to_dir=path_to_dir, train_batch_size=train_batch_size,
                         train_num_workers=train_num_workers,
                         val_batch_size=val_batch_size, valid_num_workers=valid_num_workers,
                         random_state=random_state, train_size=train_size,
                         train_transforms=train_transforms, val_transforms=val_transforms,
                         test_transforms=test_transforms, dims=dims)

    def setup(self, stage):
        super().setup(stage)
        generator = torch.Generator().manual_seed(self._random_state)

        total_samples = len(self._train_dataset)
        train_size = round(self._train_size * total_samples)
        test_size = total_samples - train_size
        assert train_size > 0
        assert test_size > 0

        assert train_size + test_size == total_samples

        self._train_dataset, self._test_datset = data.random_split(
            self._train_dataset, lengths=[train_size, test_size], generator=generator)

        self._test_datset.dataset.transformations = self.val_transforms

    def val_dataloader(self):
        return data.DataLoader(self._test_datset, batch_size=self._val_batch_size,
                               num_workers=self._valid_num_workers,
                               pin_memory=True, drop_last=False, shuffle=False)
