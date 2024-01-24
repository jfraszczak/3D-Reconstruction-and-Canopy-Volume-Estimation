from transformers import SegformerForSemanticSegmentation, BeitForSemanticSegmentation, Mask2FormerForUniversalSegmentation, MobileViTV2ForSemanticSegmentation, UperNetForSemanticSegmentation
import numpy as np
from omegaconf import DictConfig
import hydra
import os
from . import segformer
from . import mobilevitv2
from . import upernet
from .predict import predict_mask2former, predict_mobilevitv2, predict_pretrained_model, predict_beit
import evaluate
from ..data import SemanticSegmentationDataset
from torchvision import transforms
import torch

def compute_metrics(predictions: np.ndarray, labels: np.ndarray, id2label: dict) -> dict:
    '''
    Function used to compute evaluation metrics of trained model on test data.
    '''
    metric = evaluate.load("mean_iou")
    metrics = metric._compute(
        predictions=predictions,
        references=labels,
        num_labels=len(id2label),
        ignore_index=255,
        reduce_labels=True
    )

    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i + 1]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i + 1]}": v for i, v in enumerate(per_category_iou)})

    return metrics

def make_evaluation(cfg: DictConfig) -> None:
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
    
    test_data = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.test_data)
    test_annotations = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.test_annotations)
    dataset = SemanticSegmentationDataset(test_data, test_annotations, augment=False)

    predictions = []
    labels = []

    for img, segmentation_map in dataset:
        segmentation_map = segmentation_map.squeeze()
        img_pil = transforms.ToPILImage()(img)

        if cfg.model.name == "mobilevitv2" or cfg.model.name == "mobilevitv2_weighted_loss" or cfg.model.name == "beit":
            logits, upsampled_logits = predict(model, img_pil)
        else:
            logits, upsampled_logits = predict(model, cfg.model.pretrained_model, img_pil)

        prediction = (upsampled_logits.argmax(dim=1)[0])
        predictions.append(prediction.numpy())
        labels.append(segmentation_map.numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)
    metrics = compute_metrics(predictions, labels, dataset.get_id2label())
    print(metrics)
        

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    make_evaluation(cfg)

if __name__ == "__main__":
    main()
