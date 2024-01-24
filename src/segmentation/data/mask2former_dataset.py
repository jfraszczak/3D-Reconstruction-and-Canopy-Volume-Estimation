from .semantic_segmentation_dataset import SemanticSegmentationDataset
import numpy as np


class Mask2FormerDataset(SemanticSegmentationDataset):
    """
    Child class of SemanticSegmentationDataset extending parent class functionality by performing appropriate data processing for Mask2Former model.
    """
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img, segmentation_map = super().__getitem__(index)
        img = np.array(img)
        segmentation_map = np.array(segmentation_map.squeeze_())

        return img, segmentation_map
