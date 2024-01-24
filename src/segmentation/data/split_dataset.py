import os
import shutil
import random
import numpy as np
import json
from copy import copy
import hydra
from omegaconf import DictConfig
from pycocotools.coco import COCO


class DatasetSplitter:
    """
    Class used to split raw dataset into training, validation and testing subsets.

    Attributes:

    Methods:
        split(): Split data randomly into training, validation and testing subsets and save in separate directories.
    """

    def __init__(self, images_path: str, coco_path: str, output_path: str, train_size: float = 0.8, val_size: float = 0.1):
        """
        Initializes all the necessary attributes.

        Attributes:
            images_path (str): Path to the directory containing images.
            coco_path (str): Path to the COCO format json file containing annotations of aforementioned images.
            output_path (str): Path to the output directory where train, val, test subsets are to be saved.
            train_size (float): Proportion of samples to include in train dataset. It should be between 0.0 and 1.0.
            val_size (float): Proportion of samples to include in val dataset. It should be between 0.0 and 1.0.
            splits_types (list[str]): List containing names of output train, val, test subsets.
            splits_counts (list[int]): List containing number of samples in each subset.
            coco_data (dict): Data from COCO json file.
            img_ids (list[int]): List containing ids of annotated images.
            coco (COCO): Object of COCO class.

        Args:
            images_path (str): Path to the directory containing images to be segmented.
            coco_path (str): Path to the COCO format json file containing annotations of aforementioned images.
            output_path (str): Path to the output directory where train, val, test subsets are to be saved.
            train_size (float): Proportion of samples to include in train dataset. It should be between 0.0 and 1.0.
            val_size (float): Proportion of samples to include in val dataset. It should be between 0.0 and 1.0.
        """
        self.images_path = images_path
        self.coco_path = coco_path
        self.output_path = output_path
        self.train_size = train_size
        self.val_size = val_size
        self.splits_types = ["train", "val", "test"]
        self.splits_counts = [0, 0, 0]
        self.coco_data = None
        self.img_ids = []
        self.coco = COCO(self.coco_path)
        self._read_coco_data()
        self._calculate_splits_counts()

    def _read_coco_data(self) -> None:
        """
        Reads COCO format json file and filters out images without annotations.
        """
        with open(self.coco_path) as json_file:
            coco_data = json.load(json_file)
        self.coco_data = copy(coco_data)

        img_ids = set()
        for img_id in self.coco.getImgIds():
            annotations_ids = self.coco.getAnnIds(imgIds=[img_id])
            if len(annotations_ids) > 0:
                img_ids.add(img_id)
        self.img_ids = list(img_ids)[:]

    def _calculate_splits_counts(self) -> None:
        """
        Calculates number of samples to include in train, val, test subsets.
        """
        if self.train_size + self.val_size > 1.0:
            raise ValueError('Sum of train and val sizes cannot exceed 1.0')
        
        if self.train_size < 0.0:
            raise ValueError('Value of train_size cannot be smaller than 0.0')

        if self.val_size < 0.0:
            raise ValueError('Value of val_size cannot be smaller than 0.0')

        num_train = round(len(self.img_ids) * self.train_size)
        num_val = round(len(self.img_ids) * self.val_size)
        num_test = len(self.img_ids) - num_train - num_val
        self.splits_nums = [num_train, num_val, num_test]

    def _create_output_dirs(self) -> None:
        """
        Creates output directories for train, val, test data. 
        If such directories already exist they are deleted.
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.mkdir(self.output_path)

        for split_type in self.splits_types:
            set_dir = os.path.join(self.output_path, split_type)
            os.mkdir(set_dir)

    def _shuffle_images_ids(self) -> None:
        """
        Defines how images are to be shuffled before being split into separate subsets.
        """
        random.shuffle(self.img_ids)

    def split(self) -> None:
        """
        Split data randomly into training, validation and testing subsets and save in separate directories.
        Both images and COCO format json files are saved.
        """
        self._create_output_dirs()
        self._shuffle_images_ids()

        start = 0
        for num, split_type in zip(self.splits_nums, self.splits_types):
            set_images = []
            set_ids = []

            for i in range(start, start + num):
                img_id = self.img_ids[i]
                image = self.coco.loadImgs([img_id])[0]
                set_images.append(image)
                set_ids.append(img_id)
                src = os.path.join(self.images_path, image["file_name"])
                set_dir = os.path.join(self.output_path, split_type)
                dst = os.path.join(set_dir, image["file_name"])
                shutil.copy(src, dst)

            start += num

            annotations = []
            for annotation in self.coco_data["annotations"]:
                if annotation["image_id"] in set_ids:
                    annotations.append(copy(annotation))

            set_dict = {
                "licenses": self.coco_data.get("licenses", []),
                "info": self.coco_data.get("info", {}),
                "categories": self.coco_data["categories"],
                "images": set_images,
                "annotations": annotations
            }

            output_path = self.output_path + "/"
            output_file = output_path + f"annotations_{split_type}.json"
            with open(output_file, "w") as f:
                json.dump(set_dict, f, indent=2)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    dataset_path = cfg.paths.raw_data.dataset_path
    coco_path = os.path.join(dataset_path, cfg.paths.raw_data.annotations_file)
    images_path = os.path.join(dataset_path, cfg.paths.raw_data.images_dir)
    output_path = cfg.paths.dataset.dataset_path
    train_size = cfg.dataset_split.train_size
    val_size = cfg.dataset_split.val_size

    splitter = DatasetSplitter(images_path, coco_path, output_path, train_size, val_size)
    splitter.split()

if __name__ == '__main__':
    main()
