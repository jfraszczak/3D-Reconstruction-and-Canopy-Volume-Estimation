import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig
import random
import colorsys
import os
from ..data import SemanticSegmentationDataset


def show_training_dataset(train_data: str, train_annotations: str) -> None:
    """
    Visualize training dataset.

    Args:
        train_data (str): Path to the train dataset.
        train_annotations (str): Path to the annotations of training dataset.
    """
    
    # Create data loader
    dataset = SemanticSegmentationDataset(train_data, train_annotations, augment=False)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False)
    
    # Iterate through training images
    for img, segmentation_map in data_loader:
        img = img.squeeze()
        segmentation_map = segmentation_map.squeeze()

        # Print shape
        height, width = img.shape[1], img.shape[2]
        print(img.shape, segmentation_map.shape)
        print(img)

        # Show image
        img_pil = transforms.ToPILImage()(img)
        plt.imshow(img_pil)
        plt.show()

        # Show segmentation map
        segmentation_map = segmentation_map.numpy()
        color_seg = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
        palette = np.array([[0, 255, 50], [139, 69, 19], [255, 255, 255]])
        for label, color in enumerate(palette):
            color_seg[segmentation_map == label + 1, :] = color

        img_pil = np.array(img_pil) * 0.5 + color_seg * 0.5
        img_pil = img_pil.astype(np.uint8)

        plt.figure(figsize=(15, 10))
        plt.imshow(img_pil)
        plt.show()

        # Show distributions of HSV values for different label categories
        trunk_pixels = []
        canopy_pixels = []
        background_pixels = []

        trunk = []
        canopy = []
        background = []

        for i in range(height):
            for j in range(width):
                if random.random() > 0.03:
                    continue

                r, g, b = img[:, i, j].tolist()
                h, s, v = list(colorsys.rgb_to_hsv(r, g, b))
                (h, s, v) = (int(h * 360), int(s * 255), int(v * 255))

                if segmentation_map[i, j] == 1:
                    canopy_pixels.append([r, g, b])
                    canopy.append([h, s])
                elif segmentation_map[i, j] == 2:
                    trunk_pixels.append([r, g, b])
                    trunk.append([h, s])
                elif segmentation_map[i, j] == 3:
                    background_pixels.append([r, g, b])
                    background.append([h, s])

        trunk_pixels = np.array(trunk_pixels)
        canopy_pixels = np.array(canopy_pixels)
        background_pixels = np.array(background_pixels)

        ax = plt.axes(projection='3d')
        if np.shape(trunk_pixels)[0] > 0:
            ax.scatter3D(trunk_pixels[:, 0], trunk_pixels[:, 1], trunk_pixels[:, 2], c="brown")
        if np.shape(canopy_pixels)[0] > 0:
            ax.scatter3D(canopy_pixels[:, 0], canopy_pixels[:, 1], canopy_pixels[:, 2], c="green")
        if np.shape(background_pixels)[0] > 0:
            ax.scatter3D(background_pixels[:, 0], background_pixels[:, 1], background_pixels[:, 2], c="blue")
        plt.show()

        background = np.array(background)
        trunk = np.array(trunk)
        canopy = np.array(canopy)

        if np.shape(background)[0] > 0:
            plt.scatter(background[:, 0], background[:, 1], c='cyan')
        if np.shape(canopy)[0] > 0:
            plt.scatter(canopy[:, 0], canopy[:, 1], c='green')
        if np.shape(trunk)[0] > 0:
            plt.scatter(trunk[:, 0], trunk[:, 1], c='brown')
        plt.show()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_data = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.train_data)
    train_annotations = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.train_annotations)
    print(train_data)
    print(train_annotations)
    show_training_dataset(train_data, train_annotations)

if __name__ == "__main__":
    main()
