from transformers import Trainer
import torch
from torch import nn
from typing import Union, Any, Dict, Tuple, Optional, List


class Mask2FormerTrainer(Trainer):
        
        def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            loss = super().training_step(model, inputs)
            return loss.squeeze()

        def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            
            outputs = model(
                pixel_values=inputs["pixel_values"],
                mask_labels=inputs["mask_labels"],
                class_labels=inputs["class_labels"]
            )

            class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

            # Scale back to preprocessed image size - (384, 384) for all models
            masks_queries_logits = torch.nn.functional.interpolate(
                masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
            )

            # Remove the null class `[..., :-1]`
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            batch_size = class_queries_logits.shape[0]

            # Prepare labels
            labels = []
            for i in range(batch_size):
                l = 0.
                for j, class_label in enumerate(inputs["class_labels"][i].tolist()):
                    l += inputs["mask_labels"][i][j] * (class_label - 1)
                labels.append(l)
            labels = torch.stack(labels, dim=0)

            return outputs.loss.squeeze_().detach(), segmentation.detach(), labels
        