from transformers import UperNetForSemanticSegmentation
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SemanticSegmenterOutput


class UperNetForSemanticSegmentationWeightedCrossEntropy(UperNetForSemanticSegmentation):

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[tuple, SemanticSegmenterOutput]:
            r"""
            labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
                Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

            Returns:

            Examples:
            ```python
            >>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
            >>> from PIL import Image
            >>> from huggingface_hub import hf_hub_download

            >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
            >>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

            >>> filepath = hf_hub_download(
            ...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
            ... )
            >>> image = Image.open(filepath).convert("RGB")

            >>> inputs = image_processor(images=image, return_tensors="pt")

            >>> outputs = model(**inputs)

            >>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
            >>> list(logits.shape)
            [1, 150, 512, 512]
            ```"""

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

            outputs = self.backbone.forward_with_filtered_kwargs(
                pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            features = outputs.feature_maps

            logits = self.decode_head(features)
            logits = nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

            auxiliary_logits = None
            if self.auxiliary_head is not None:
                auxiliary_logits = self.auxiliary_head(features)
                auxiliary_logits = nn.functional.interpolate(
                    auxiliary_logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
                )

            loss = None
            if labels is not None:
                if self.config.num_labels == 1:
                    raise ValueError("The number of labels should be greater than one")
                else:
                    # Compute weights
                    class_count = []
                    for c in range(self.config.num_labels):
                        class_count.append(((labels == c).float()).sum())

                    class_count = torch.tensor(class_count, dtype=torch.float, device=pixel_values.get_device())
                    n_samples = labels.shape[0] * labels.shape[1] * labels.shape[2]
                    weights = n_samples / (class_count * self.config.num_labels)

                    # compute weighted loss
                    loss_fct_weighted = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index, weight=weights)
                    loss_fct = CrossEntropyLoss(ignore_index=self.config.loss_ignore_index)
                    loss = loss_fct_weighted(logits, labels)
                    if auxiliary_logits is not None:
                        auxiliary_loss = loss_fct(auxiliary_logits, labels)
                        loss += self.config.auxiliary_loss_weight * auxiliary_loss

            if not return_dict:
                if output_hidden_states:
                    output = (logits,) + outputs[1:]
                else:
                    output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return SemanticSegmenterOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
