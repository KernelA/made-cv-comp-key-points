import os
import logging
from typing import Iterable, List, Union
import logging
import ast
import shutil

import torch
from tqdm import trange
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

        self._dump_dir = None

        landmarks_data = pd.read_csv(
            csv_landmarks_date, sep="\t", index_col="file_name", engine="c")

        self._image_names = landmarks_data.index.tolist()

        if self._is_train:
            self._landmarks_points = torch.from_numpy(landmarks_data.to_numpy()).reshape(
                len(self._image_names), len(landmarks_data.columns) // 2, 2)
        else:
            self._point_indices = torch.LongTensor(
                [ast.literal_eval(indices) for indices in landmarks_data["point_index_list"]])

    def index2img_name(self, index: Union[int, Iterable[int]]) -> Union[str, List[str]]:
        if isinstance(index, int):
            return self._image_names[index]
        else:
            return [self._image_names[i] for i in index]

    def __len__(self):
        return len(self._image_names)

    def precompute(self, dump_dir: str):
        self._logger.info("Precompute data and save tensor to %s", dump_dir)

        for i in trange(len(self)):
            data = self[i]
            path_to_dump = os.path.join(dump_dir, data["image_name"])
            torch.save(data, self._get_dump_path(path_to_dump))

        self._dump_dir = dump_dir

    def _get_dump_path(self, path):
        return os.path.splitext(path)[0] + ".pth"

    def __getitem__(self, index):
        if self._dump_dir is not None:
            dump_path = os.path.join(self._dump_dir, self._image_names[index])
            return torch.load(self._get_dump_path(dump_path))
        else:
            image = read_image(os.path.join(self._image_dir, self._image_names[index]))

            height, width = image.shape[-2:]

            if self.transformations is not None:
                image = self.transformations(image)

            data = {"image": image, "image_name": self._image_names[index]}

            if self._is_train:
                data["norm_landmarks"] = normalize_landmarks(
                    self._landmarks_points[index], width, height)
            else:
                data["point_indices"] = self._point_indices[index]
                data["img_height"] = height
                data["img_width"] = width

            return data


class FullLandmarkDataModule(pl.LightningDataModule):
    def __init__(self, *, path_to_dir: str, train_batch_size: int,
                 train_num_workers: int, val_batch_size: int, valid_num_workers: int, random_state: int,
                 train_size: float = 1, precompute_data: bool, train_transforms=None,
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
        self._precompute_data = precompute_data
        self._logger = logging.getLogger("kp.datamodule")

    def setup(self, stage=None):
        general_dataset = LandMarkDatset(
            path_to_dir=self._data_dir, is_train=True, transformations=self.train_transforms)

        self._train_dataset = general_dataset

        if self._precompute_data:
            dump_dir = os.path.join(self._data_dir, "train_dump")

            if os.path.isdir(dump_dir):
                shutil.rmtree(dump_dir)
            os.makedirs(dump_dir, exist_ok=True)

            self._train_dataset.precompute(dump_dir)

    def train_dataloader(self):
        return data.DataLoader(self._train_dataset, shuffle=True, drop_last=True,
                               batch_size=self._train_batch_size,
                               num_workers=self._train_num_workers, pin_memory=True)


class TrainTestLandmarkDataModule(FullLandmarkDataModule):
    def __init__(self, *, path_to_dir: str, train_batch_size: int,
                 train_num_workers: int, val_batch_size: int, valid_num_workers: int,
                 random_state: int, train_size: float, precompute_data: bool,
                 train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(path_to_dir=path_to_dir,
                         train_batch_size=train_batch_size,
                         train_num_workers=train_num_workers,
                         precompute_data=precompute_data,
                         val_batch_size=val_batch_size,
                         valid_num_workers=valid_num_workers,
                         random_state=random_state,
                         train_size=train_size,
                         train_transforms=train_transforms,
                         val_transforms=val_transforms,
                         test_transforms=test_transforms,
                         dims=dims)

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

        if self._precompute_data:
            dump_dir = os.path.join(self._data_dir, "test_dump")

            if os.path.isdir(dump_dir):
                shutil.rmtree(dump_dir)
            os.makedirs(dump_dir, exist_ok=True)

            self._test_datset.precompute(dump_dir)

    def val_dataloader(self):
        return data.DataLoader(self._test_datset, batch_size=self._val_batch_size,
                               num_workers=self._valid_num_workers,
                               pin_memory=True, drop_last=False, shuffle=False)
