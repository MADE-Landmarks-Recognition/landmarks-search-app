import os

import pandas as pd
from sklearn import preprocessing

from .base_dataset import BaseDataset


def img_path_from_id(id, data_dir):
    img_path = os.path.join(data_dir, id[0], id[1], id[2], f"{id}.jpg")
    return img_path


class GLDv2Dataset(BaseDataset):
    def __init__(
        self,
        data_dir,
        csv_path,
        test_size,
        mode,
        transform=None,
        load_super_labels=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.mode = mode
        self.transform = transform
        self.load_super_labels = load_super_labels
        self.landmark_id_encoder = preprocessing.LabelEncoder()

        df = pd.read_csv(csv_path)
        df["landmark_id"] = self.landmark_id_encoder.fit_transform(df["landmark_id"])
        paths = [img_path_from_id(img_id, data_dir) for img_id in df["id"].values]
        labels = df["landmark_id"].values
        self.classes_ = self.landmark_id_encoder.classes_
        with open("classes.txt", "w") as f:
            f.write("\n".join(map(str, self.classes_)))

        sorted_lb = list(sorted(set(labels)))
        if mode == "train":
            set_labels = set(sorted_lb[: int(-test_size * len(sorted_lb))])
        elif mode == "test":
            set_labels = set(sorted_lb[int(-test_size * len(sorted_lb)) :])
        elif mode == "all":
            set_labels = sorted_lb

        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            if lb in set_labels:
                self.paths.append(pth)
                self.labels.append(lb)

        self.super_labels = None

        self.get_instance_dict()


class GLDv2DatasetClassificationSplits(BaseDataset):
    def __init__(
        self,
        data_dir,
        csv_path,
        test_size,
        mode,
        transform=None,
        load_super_labels=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.mode = mode
        self.transform = transform
        self.load_super_labels = load_super_labels
        self.landmark_id_encoder = preprocessing.LabelEncoder()

        df = pd.read_csv(csv_path)
        df["landmark_id"] = self.landmark_id_encoder.fit_transform(df["landmark_id"])
        self.classes_ = self.landmark_id_encoder.classes_
        with open("classes.txt", "w") as f:
            f.write("\n".join(map(str, self.classes_)))

        df = df[df.stage == mode]
        self.paths = [img_path_from_id(img_id, data_dir) for img_id in df["id"].values]
        self.labels = df["landmark_id"].values
        self.super_labels = None

        self.get_instance_dict()
