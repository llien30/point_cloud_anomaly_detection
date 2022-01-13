import copy
import glob
import json
import os
import random

# from .visualize import vis_points_3d
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils import data

from .load_obj import loadOBJ
from .sampling import fartherst_point_sampling


class ShapeNetDataset(data.Dataset):
    def __init__(self, csv_file, sampling="fps", n_point=2000):
        super().__init__()
        assert sampling.lower() in [
            "fps",
            "random",
            "order",
        ], "The sampling method must be 'fps','random', or 'order'!"

        self.df = pd.read_csv(csv_file)
        self.n_point = n_point
        self.sampling = sampling

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["path"]
        point = loadOBJ(path)
        point = np.array(point)

        # vis_point = torch.tensor(point)
        # vis_points_3d(vis_point, f"./{idx}_original.png")

        point_idx = [i for i in range(point.shape[0])]
        point_idx = np.array(point_idx)

        # points sampling
        if self.sampling == "fps":
            point_idx = fartherst_point_sampling(point, self.n_point)
            point = point[point_idx]

        elif self.sampling == "random":
            if point.shape[0] >= self.n_point:
                point_choice = np.random.choice(point_idx, self.n_point, replace=False)
            else:
                point_choice = np.random.choice(point_idx, self.n_point, replace=True)
            point = point[point_choice, :]

        elif self.sampling == "order":
            point = point[: self.n_point]

        # point = torch.tensor(point)
        # vis_points_3d(point, f"./{idx}_sampling.png")

        label = self.df.iloc[idx]["label"]

        sample = {
            "data": point,
            "label": label,
            "name": path[31:39],
            "path": path,
        }

        return sample


class ShapeNeth5pyDataset(data.Dataset):
    def __init__(
        self,
        root_path: str,
        split: str,
        normal_class: list,
        abnormal_class: list,
        n_point: int,
        random_rotate: bool = False,
        random_jitter: bool = False,
        random_translate: bool = False,
    ) -> None:
        split = split.lower()
        assert split in [
            "train",
            "test",
            "validation",
        ], "split must be train, test, or validation"

        self.n_point = n_point
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        normal_data_paths = []
        for n_class in normal_class:
            normal_data_paths.append(os.path.join(root_path, split, f"{n_class}.h5"))
        normal_data, normal_name = self.load_h5py(normal_data_paths)

        normal_data = np.concatenate(normal_data, axis=0)
        normal_name = np.concatenate(normal_name, axis=0)

        # 正常データをシャッフルする
        if split == "test":
            tmp = list(zip(normal_data, normal_name))
            random.shuffle(tmp)
            normal_data, normal_name = zip(*tmp)

        self.normal_data = normal_data
        self.normal_name = normal_name
        self.normal_label = [0] * len(self.normal_data)

        if split == "train":
            self.abnormal_data = np.array([])
            self.abnormal_name = np.array([])
            self.abnormal_label = []

        else:
            abnormal_data_paths = []
            for a_class in abnormal_class:
                abnormal_data_paths.append(
                    os.path.join(root_path, split, f"{a_class}.h5")
                )
            abnormal_data, abnormal_name = self.load_h5py(abnormal_data_paths)

            # 異常データをシャッフルする
            abnormal_data = np.concatenate(abnormal_data, axis=0)
            abnormal_name = np.concatenate(abnormal_name, axis=0)
            tmp = list(zip(abnormal_data, abnormal_name))
            random.shuffle(tmp)
            abnormal_data, abnormal_name = zip(*tmp)

            self.abnormal_data = abnormal_data
            self.abnormal_name = abnormal_name

            self.abnormal_label = [1] * len(self.abnormal_data)

        if split == "train":
            self.data = self.normal_data
            self.name = self.normal_name
            self.label = self.normal_label

        else:
            len_data = min(len(self.normal_data), len(self.abnormal_data))
            self.data = np.concatenate(
                [self.normal_data[:len_data], self.abnormal_data[:len_data]], axis=0
            )
            self.name = np.concatenate(
                [self.normal_name[:len_data], self.abnormal_name[:len_data]], axis=0
            )
            self.label = self.normal_label[:len_data] + self.abnormal_label[:len_data]

    def load_h5py(self, path: list) -> Tuple[list, list]:
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, "r+")
            data = f["data"][:].astype("float32")
            label = f["label"][:]
            f.close()
            all_data.append(data)
            all_label.append(label)
            print(f"{label[0]} : {len(data)}")
        return all_data, all_label

    def __getitem__(self, idx: int) -> dict:

        point_set = self.data[idx][: self.n_point]
        name = self.name[idx]
        label = self.label[idx]

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        point_set = change2positive(point_set)
        point_set = uniform_size(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)

        return {"data": point_set, "label": label, "name": name}

    def __len__(self):
        return self.data.shape[0]


def change2positive(pointcloud):
    min_x = min(pointcloud[:, 0])
    pointcloud[:, 0] -= np.array([min_x] * pointcloud.shape[0])
    min_y = min(pointcloud[:, 1])
    pointcloud[:, 1] -= np.array([min_y] * pointcloud.shape[0])
    min_z = min(pointcloud[:, 2])
    pointcloud[:, 2] -= np.array([min_z] * pointcloud.shape[0])

    return pointcloud


def uniform_size(pointcloud):
    min_x = min(pointcloud[:, 0])
    max_x = max(pointcloud[:, 0])
    min_y = min(pointcloud[:, 1])
    max_y = max(pointcloud[:, 1])
    min_z = min(pointcloud[:, 2])
    max_z = max(pointcloud[:, 2])
    pointcloud[:, 0] /= max_x - min_x
    pointcloud[:, 1] /= max_y - min_y
    pointcloud[:, 2] /= max_z - min_z

    return pointcloud


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
        "float32"
    )
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    pointcloud = copy.deepcopy(pointcloud)
    N, C = pointcloud.shape
    # pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    # pointcloud += np.random.normal(0, 0.02, size=pointcloud.shape)
    x_min = min(pointcloud[:, 0])
    x_max = max(pointcloud[:, 0])
    pointcloud[:, 0] -= np.array([x_min] * pointcloud.shape[0])
    y_min = min(pointcloud[:, 1])
    y_max = max(pointcloud[:, 1])
    pointcloud[:, 1] -= np.array([y_min] * pointcloud.shape[0])
    z_min = min(pointcloud[:, 2])
    z_max = max(pointcloud[:, 2])
    pointcloud[:, 2] -= np.array([z_min] * pointcloud.shape[0])
    x_jitter = np.random.normal(0, (x_max - x_min) * 0.01, size=(pointcloud.shape[0]))
    y_jitter = np.random.normal(0, (y_max - y_min) * 0.01, size=(pointcloud.shape[0]))
    z_jitter = np.random.normal(0, (z_max - z_min) * 0.01, size=(pointcloud.shape[0]))
    pointcloud[:, 0] += x_jitter
    pointcloud[:, 1] += y_jitter
    pointcloud[:, 2] += z_jitter

    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(
        rotation_matrix
    )  # random rotation (x,z)
    return pointcloud


class Dataseth5py(data.Dataset):
    def __init__(
        self,
        root,
        dataset_name="shapenetcorev2",
        num_points=2048,
        split="train",
        load_name=False,
        random_rotate=False,
        random_jitter=False,
        random_translate=False,
    ):

        assert dataset_name.lower() in [
            "shapenetcorev2",
            "shapenetpart",
            "modelnet10",
            "modelnet40",
        ]
        assert num_points <= 2048

        if dataset_name in ["shapenetpart", "shapenetcorev2"]:
            assert split.lower() in ["train", "test", "val", "trainval", "all"]
        else:
            assert split.lower() in ["train", "test", "all"]

        self.root = os.path.join(root, dataset_name + "*hdf5_2048")
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        self.path_h5py_all = []
        self.path_json_all = []
        if self.split in ["train", "trainval", "all"]:
            self.get_path("train")
        if self.dataset_name in ["shapenetpart", "shapenetcorev2"]:
            if self.split in ["val", "trainval", "all"]:
                self.get_path("val")
        if self.split in ["test", "all"]:
            self.get_path("test")

        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)  # load label name

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)

    def get_path(self, type):
        path_h5py = os.path.join(self.root, "*%s*.h5" % type)
        self.path_h5py_all += glob.glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, "%s*_id2name.json" % type)
            self.path_json_all += glob.glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, "r+")
            data = f["data"][:].astype("float32")
            label = f["label"][:].astype("int64")
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, "r+")
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][: self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        if self.load_name:
            return {"data": point_set, "label": label, "name": name}
        else:
            return {"data": point_set, "label": label}

    def __len__(self):
        return self.data.shape[0]
