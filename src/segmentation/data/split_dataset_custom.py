import random
import hydra
from omegaconf import DictConfig
import os
from .split_dataset import DatasetSplitter


class CustomDataSplitter(DatasetSplitter):

    def _calculate_splits_counts(self) -> None:
        """
        Calculates number of samples to include in train, val, test subsets.
        Assumes the last 25 images to be allocated in test subset as they are
        from the other side of the row and are not overlapping with remaining images.
        It is supposed to make evaluation more reliable and less prone to 
        information leakage.
        """
        if self.train_size + self.val_size > 1.0:
            raise ValueError('Sum of train and val sizes cannot exceed 1.0')
        
        if self.train_size < 0.0:
            raise ValueError('Value of train_size cannot be smaller than 0.0')

        if self.val_size < 0.0:
            raise ValueError('Value of val_size cannot be smaller than 0.0')

        num_train = round((len(self.img_ids) - 25) * self.train_size)
        num_val = round((len(self.img_ids) - 25) * self.val_size)
        num_test = 25
        self.splits_nums = [num_train, num_val, num_test]

    def _shuffle_images_ids(self) -> None:
        """
        Defines how images are to be shuffled before being split into separate subsets.
        """
        train_val_ids = self.img_ids[:-25]
        random.shuffle(train_val_ids)
        self.img_ids = train_val_ids + self.img_ids[-25:]

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    dataset_path = cfg.paths.raw_data.dataset_path
    coco_path = os.path.join(dataset_path, cfg.paths.raw_data.annotations_file)
    images_path = os.path.join(dataset_path, cfg.paths.raw_data.images_dir)
    output_path = cfg.paths.dataset.dataset_path
    train_size = cfg.dataset_split.train_size
    val_size = cfg.dataset_split.val_size

    splitter = CustomDataSplitter(images_path, coco_path, output_path, train_size, val_size)
    splitter.split()

if __name__ == '__main__':
    main()
