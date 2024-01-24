import os
import torch
from torch.utils import data
import PIL
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
from torchvision.transforms import v2


class SemanticSegmentationDataset(data.Dataset):
    """
    Implementation of PyTorch Dataset class storing the samples and their corresponding labels for image segmentation task.

    Attributes:
        images_path (str): Path to the directory containing images to be segmented.
        coco (COCO): Object of COCO class for reading annotations.
        augment (bool): Boolean flag indicating whether to perform augmentation of data or not.
        jitter (transforms.ColorJitter): Object of ColorJitter class used for color augmentation. It randomly changes the brightness, contrast, saturation and hue of an image.
        transform (v2.Transform): Object of v2.Transform class used for geometrical transformation. It performs random horiontal flip and rotation.

    Methods:
        __len__(): Returns number of images in the dataset.
        __get__(): Returns index-th image and it's segmentation map as torch Tensors.
    """

    def __init__(self, images_path: str, coco_path: str, augment: bool=False) -> None:
        """
        Initializes all the necessary attributes and reads data from COCO format json file.

        Args:
            images_path (str): Path to the directory containing images to be segmented.
            coco_path (str): Path to the COCO format json file containing annotations of aforementioned images.
            augment (bool): Boolean flag indicating whether to perform augmentation of data or not.
        """
        self.images_path = images_path
        self.coco = COCO(coco_path)
        self.augment = augment
        self.jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15)
        ])

    def _get_image_shape(self, img_id: int) -> tuple[int, int]:
        """
        Returns shape (height, width) of image with specified id.

        Args:
            img_id (int): Id of the image whose shape is to be returned.

        Returns:
            tuple[int, int]: Shape (height, width) of the image. 
        """
        width = self.coco.loadImgs([img_id])[0]["width"]
        height = self.coco.loadImgs([img_id])[0]["height"]

        return (height, width)

    def _get_image(self, img_id: int) -> torch.Tensor:
        """
        Returns tensor representing image with specified id.

        Args:
            img_id (int): Id of the image which is to be returned.

        Returns:
            torch.Tensor: Tensor representing image of shape (3, height, width).
        """
        name = self.coco.loadImgs([img_id])[0]["file_name"]
        img = PIL.Image.open(os.path.join(self.images_path, name))
        img = transforms.ToTensor()(img)

        return img

    def _get_segmentation_map(self, img_id: int) -> torch.Tensor:
        """
        Returns tensor representing segmentation map with the same shape as image specified by img_id.

        Args:
            img_id (int): Id of the image whose segmentation map is to be returned.

        Returns:
            torch.Tensor: Tensor representing segmentation map of shape (height, width).
        """
        annotations_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(annotations_ids)
        segmentation_map = np.zeros(self._get_image_shape(img_id))

        for annotation in annotations:
            mask = self.coco.annToMask(annotation)
            segmentation_map[segmentation_map == 0] += (mask * annotation['category_id'])[segmentation_map == 0]
        segmentation_map[segmentation_map == 0] = len(self.coco.dataset['categories']) + 1  # Last label for background
        segmentation_map = transforms.ToTensor()(segmentation_map)

        return segmentation_map
    
    def _apply_augmentation(self, img: torch.Tensor, segmentation_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies color augmentation by randomly changing the brightness, contrast, saturation and hue of an image as well as 
        geometrical transformation such as random horiontal flip and rotation.

        Args:
            img (torch.Tensor): Image to be augmented.
            segmentation_map (torch.Tensor): Corresponding segmentation map.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Augmented image of shape (3, height, width) and it's segmentation map of shape (height, width).
        """
        img = self.jitter(img)
        stacked = torch.cat([img, segmentation_map], dim=0)
        stacked = self.transform(stacked)
        img = stacked[:-1]
        segmentation_map = stacked[-1]

        return img, segmentation_map
    
    def get_id2label(self) -> dict:
        """
        Returns mapping from ids to labels, where 'background' has the last id 
        for example {1: 'canopy', 2: 'trunk', 3: 'background'}.

        Returns:
            dict: Mapping from ids to labels.
        """
        id2label = dict()
        for category in self.coco.dataset['categories']:
            id2label[category['id']] = category['name'].lower()
        id2label[len(self.coco.dataset['categories']) + 1] = 'background'
        
        return id2label
    
    def __len__(self) -> int:
        """
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.coco.getImgIds())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns index-th image and it's segmentation map as torch Tensors.

        Args:
            index (int): Index of image which is to be returned.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Image of shape (3, height, width) with rescaled values between 0-1 and it's segmentation map of shape (height, width).
        """
        img_id = self.coco.getImgIds()[index]
        img = self._get_image(img_id)
        segmentation_map = self._get_segmentation_map(img_id)

        if self.augment:
            img, segmentation_map = self._apply_augmentation(img, segmentation_map)

        return img, segmentation_map
