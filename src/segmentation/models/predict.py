from transformers import SegformerForSemanticSegmentation, BeitForSemanticSegmentation, Mask2FormerForUniversalSegmentation, MobileViTV2ForSemanticSegmentation, UperNetForSemanticSegmentation
import torch
import numpy as np
from matplotlib import pyplot as plt
import PIL
from omegaconf import DictConfig
import hydra
import os
from . import segformer
from . import mobilevitv2
from . import upernet
import transformers


def predict_pretrained_model(model: transformers.PreTrainedModel, pretrained_model: str, image: PIL.Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns predicted logits of shape (batch_size, num_labels, height, width) and 
    upsampled_logits after interpolation in order to match size of input image.

    Args:
        model (transformers.PreTrainedModel): Object of PreTrainedModel class which is to perform a prediction. It might represent for example Segformer or UPerNet.
        pretrained_model (str): Name of pretrained model according to which appropriate image processor is instantiated.
        image (PIL.Image.Image): Image which is to be segmented.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Pair of predictied logits and upsampled_logits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = transformers.AutoImageProcessor.from_pretrained(pretrained_model, do_center_crop=False)
    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(logits,
                                                       size=image.size[::-1],
                                                       mode='bilinear',
                                                       align_corners=False)
    return logits, upsampled_logits


def predict_mobilevitv2(model: transformers.MobileViTV2ForSemanticSegmentation, image: PIL.Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns predicted logits of shape (batch_size, num_labels, height, width) and 
    upsampled_logits after interpolation in order to match size of input image.

    Args:
        model (transformers.MobileViTV2ForSemanticSegmentation): Object of MobileViTV2ForSemanticSegmentation class which is to perform a prediction.
        image (PIL.Image.Image): Image which is to be segmented.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Pair of predictied logits and upsampled_logits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = transformers.SegformerImageProcessor(size=384, do_center_crop=False)
    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(logits,
                                                       size=image.size[::-1],
                                                       mode='bilinear',
                                                       align_corners=False)
    return logits, upsampled_logits

def predict_beit(model: transformers.BeitForSemanticSegmentation, image: PIL.Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns predicted logits of shape (batch_size, num_labels, height, width) and 
    upsampled_logits after interpolation in order to match size of input image.

    Args:
        model (transformers.BeitForSemanticSegmentation): Object of BeitForSemanticSegmentation class which is to perform a prediction.
        image (PIL.Image.Image): Image which is to be segmented.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Pair of predictied logits and upsampled_logits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = transformers.SegformerImageProcessor(size=384, do_center_crop=False)
    encoding = image_processor(image, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(logits,
                                                       size=image.size[::-1],
                                                       mode='bilinear',
                                                       align_corners=False)
    return logits, upsampled_logits

def predict_mask2former(model: Mask2FormerForUniversalSegmentation, pretrained_model: str, image: PIL.Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns predicted logits of shape (batch_size, num_labels, height, width) and 
    upsampled_logits after interpolation in order to match size of input image.

    Args:
        model (transformers.Mask2FormerForUniversalSegmentation): Object of Mask2FormerForUniversalSegmentation class which is to perform a prediction.
        pretrained_model (str): Name of pretrained model according to which appropriate image processor is instantiated.
        image (PIL.Image.Image): Image which is to be segmented.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Pair of predictied logits and upsampled_logits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = transformers.AutoImageProcessor.from_pretrained(pretrained_model)
    inputs = image_processor(image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    outputs = model(pixel_values=pixel_values)

    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # Scale back to preprocessed image size - (384, 384) for all models
    masks_queries_logits = torch.nn.functional.interpolate(
        masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
    )

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

    upsampled_logits = torch.nn.functional.interpolate(segmentation,
                                                       size=image.size[::-1],
                                                       mode='bilinear',
                                                       align_corners=False)
    return segmentation, upsampled_logits


def visualise_prediction(image: PIL.Image.Image, upsampled_logits: torch.Tensor) -> None:
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

    color_palette = [[0, 255, 50], [139, 69, 19], [255, 255, 255]]
    for label, color in enumerate(color_palette):
        color_seg[seg == label, :] = color

    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.show()

def visualise_predictions_on_val_dataset(cfg: DictConfig) -> None:
    if cfg.model.name == "segformer":
        model = SegformerForSemanticSegmentation
        predict = predict_pretrained_model
    elif cfg.model.name == "segformer_weighted_loss":
        model = segformer.SegformerForSemanticSegmentationWeightedCrossEntropy
        predict = predict_pretrained_model
    elif cfg.model.name == "beit":
        model = BeitForSemanticSegmentation
        predict = predict_beit
    elif cfg.model.name == "mask2former":
        model = Mask2FormerForUniversalSegmentation
        predict = predict_mask2former
    elif cfg.model.name == "upernet":
        model = UperNetForSemanticSegmentation
        predict = predict_pretrained_model
    elif cfg.model.name == "upernet_weighted_loss":
        model = upernet.UperNetForSemanticSegmentationWeightedCrossEntropy
        predict = predict_pretrained_model
    elif cfg.model.name == "mobilevitv2":
        model = MobileViTV2ForSemanticSegmentation
        predict = predict_mobilevitv2
    elif cfg.model.name == "mobilevitv2_weighted_loss":
        model = mobilevitv2.MobileViTV2ForSemanticSegmentationWeightedCrossEntropy
        predict = predict_mobilevitv2
    else:
        raise Exception("Specified model is wrong. "
                        "Available options are: segformer, segformer_weighted_loss, beit, mask2former, upernet, upernet_weighted_loss, mobilevitv2, mobilevitv2_weighted_loss")

    model = model.from_pretrained(cfg.model.best_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    val_images_path = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.val_data)
    for img in os.listdir(val_images_path):
        img_path = os.path.join(val_images_path, img)
        image = PIL.Image.open(img_path)

        if cfg.model.name == "mobilevitv2" or cfg.model.name == "mobilevitv2_weighted_loss" or cfg.model.name == "beit":
            logits, upsampled_logits = predict(model, image)
        else:
            logits, upsampled_logits = predict(model, cfg.model.pretrained_model, image)

        plt.imshow(logits.argmax(dim=1)[0])
        plt.show()

        visualise_prediction(image, upsampled_logits)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    visualise_predictions_on_val_dataset(cfg)

if __name__ == "__main__":
    main()
