from .semantic_segmentation_dataset import SemanticSegmentationDataset
import transformers


class MobileVitDataset(SemanticSegmentationDataset):
    """
    Child class of SemanticSegmentationDataset extending parent class functionality by performing appropriate data processing for pretrained, 
    MobileVit segmentation model. 
    SegformerImageProcessor has been used as MobileVitImageProcessor class does not provide functionality to preprocess segmentation map.

    Attributes:
        images_path (str): Path to the directory containing images to be segmented.
        coco (COCO): Object of COCO class for reading annotations.
        augment (bool): Boolean flag indicating whether to perform augmentation of data or not.
        jitter (transforms.ColorJitter): Object of ColorJitter class used for color augmentation. It randomly changes the brightness, contrast, saturation and hue of an image.
        transform (v2.Transform): Object of v2.Transform class used for geometrical transformation. It performs random horiontal flip and rotation.
        image_processor (transformers.SegformerImageProcessor): Image processor objects used for data processing.

    Methods:
        __len__(): Returns number of images in the dataset.
        __get__(): Returns index-th image and it's segmentation map as torch Tensors.
    """
    def __init__(self, images_path: str, coco_path: str, augment: bool=False) -> None:
        super().__init__(images_path, coco_path, augment)
        self.image_processor = transformers.SegformerImageProcessor(size=384, do_reduce_labels=True, do_center_crop=False, do_rescale=False)

    def __getitem__(self, index: int) -> transformers.image_processing_utils.BatchFeature:
        """
        Returns index-th image and it's segmentation map as output of specific image processor.

        Args:
            index (int): Index of image which is to be returned.
        
        Returns:
            transformers.image_processing_utils.BatchFeature: Output of specific image processor.
        """
        img, segmentation_map = super().__getitem__(index)
        encoded_inputs = self.image_processor(img, segmentation_map, return_tensors="pt")
        for k, _ in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs
