from transformers import SegformerForSemanticSegmentation, BeitForSemanticSegmentation, Mask2FormerForUniversalSegmentation, UperNetForSemanticSegmentation, MobileViTV2ForSemanticSegmentation, TrainingArguments, Trainer, AutoImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import EvalPrediction
import torch
from torch import nn
from torch.utils.data import Dataset
import evaluate
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import hydra
import os
from typing import Callable
import functools
from ..data import PretrainedModelDataset, Mask2FormerDataset, MobileVitDataset, BeitDataset
from . import segformer
from . import mask2former
from . import mobilevitv2
from . import upernet

torch.cuda.empty_cache()

def data_collator_fn(batch, pretrained_model: str):
    '''
    Function used to collate batch of data in case of mask2former data loader.
    Pretrained model is specified using partial functions.
    '''
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]

    image_processor = AutoImageProcessor.from_pretrained(pretrained_model, do_reduce_labels=True, do_rescale=False, ignore_index=255)
    batch = image_processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )
    
    return batch

metric = evaluate.load("mean_iou")

def compute_metrics_fn(eval_pred: EvalPrediction, id2label: dict) -> dict:
    '''
    Function used to perform evaluation of trained model on validation data.
    '''

    with torch.no_grad():
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        logits_tensor = torch.from_numpy(logits)

        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().numpy()

        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=False
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i + 1]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i + 1]}": v for i, v in enumerate(per_category_iou)})

        return metrics
    
def train(model: PreTrainedModel,
          train_dataset: Dataset,
          val_dataset: Dataset,
          trainer: Trainer,
          data_collator: Callable,
          pretrained_model: str,
          lr: float,
          batch_size: int,
          epochs: int) -> None:
    
    '''
    Train specified model using huggingface Trainer API.
    '''
    # Specify training arguments
    training_args = TrainingArguments(
        'finetuned_model',
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        save_strategy="epoch",
        save_steps=1,
        logging_steps=1,
        evaluation_strategy="epoch",
        eval_steps=1,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model='eval_mean_iou'
    )

    # Define model
    id2label = train_dataset.get_id2label()
    label2id = {label: id for id, label in id2label.items()}
    model = model.from_pretrained(pretrained_model,
                                  id2label=id2label,
                                  label2id=label2id,
                                  ignore_mismatched_sizes=True)

    # Run training
    trainer = trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=functools.partial(compute_metrics_fn, id2label=id2label),
        data_collator=data_collator
    )
    trainer.train()

def run_training(cfg: DictConfig) -> None:
    images_train = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.train_data)
    annotations_train = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.train_annotations)

    images_val = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.val_data)
    annotations_val = os.path.join(cfg.paths.dataset.dataset_path, cfg.paths.dataset.val_annotations)

    args_train = (images_train, annotations_train, cfg.model.pretrained_model)
    args_val = (images_val, annotations_val, cfg.model.pretrained_model)

    trainer = Trainer
    data_collator = None

    if cfg.model.name == "segformer":
        model = SegformerForSemanticSegmentation
        dataset = PretrainedModelDataset
    elif cfg.model.name == "segformer_weighted_loss":
        model = segformer.SegformerForSemanticSegmentationWeightedCrossEntropy
        dataset = PretrainedModelDataset
    elif cfg.model.name == "beit":
        model = BeitForSemanticSegmentation
        dataset = BeitDataset
        args_train = (images_train, annotations_train)
        args_val = (images_val, annotations_val)
    elif cfg.model.name == "mask2former":
        model = Mask2FormerForUniversalSegmentation
        dataset = Mask2FormerDataset
        trainer = mask2former.Mask2FormerTrainer
        data_collator = functools.partial(data_collator_fn, pretrained_model=cfg.model.pretrained_model)
        args_train = (images_train, annotations_train)
        args_val = (images_val, annotations_val)
    elif cfg.model.name == "upernet":
        model = UperNetForSemanticSegmentation
        dataset = PretrainedModelDataset
    elif cfg.model.name == "upernet_weighted_loss":
        model = upernet.UperNetForSemanticSegmentationWeightedCrossEntropy
        dataset = PretrainedModelDataset
    elif cfg.model.name == "mobilevitv2":
        model = MobileViTV2ForSemanticSegmentation
        dataset = MobileVitDataset
        args_train = (images_train, annotations_train)
        args_val = (images_val, annotations_val)
    elif cfg.model.name == "mobilevitv2_weighted_loss":
        model = mobilevitv2.MobileViTV2ForSemanticSegmentationWeightedCrossEntropy
        dataset = MobileVitDataset
        args_train = (images_train, annotations_train)
        args_val = (images_val, annotations_val)
    else:
        raise Exception("Specified model is wrong. "
                "Available options are: segformer, segformer_weighted_loss, beit, mask2former, upernet, upernet_weighted_loss, mobilevitv2, mobilevitv2_weighted_loss")

    train_dataset = dataset(*args_train, augment=True)
    val_dataset = dataset(*args_val, augment=False)

    train(
        model,
        train_dataset,
        val_dataset,
        trainer,
        data_collator,
        cfg.model.pretrained_model,
        lr=cfg.params.lr,
        batch_size=cfg.params.batch_size,
        epochs=cfg.params.epochs
    )


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)

if __name__ == "__main__":
    main()
