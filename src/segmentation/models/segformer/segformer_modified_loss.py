from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput

from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


class SegformerForSemanticSegmentationWeightedCrossEntropy(SegformerForSemanticSegmentation):

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states)

        loss = None
        if labels is not None:
            # upsample logits to the images' original size
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if self.config.num_labels > 1:
                
                # Compute weights
                class_count = []
                for c in range(self.config.num_labels):
                    class_count.append(((labels == c).float()).sum())

                class_count = torch.tensor(class_count, dtype=torch.float, device=pixel_values.get_device())
                n_samples = labels.shape[0] * labels.shape[1] * labels.shape[2]
                weights = n_samples / (class_count * self.config.num_labels)

                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index,
                                            weight=weights)
                loss = loss_fct(upsampled_logits, labels)
            elif self.config.num_labels == 1:
                valid_mask = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss_fct = BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(upsampled_logits.squeeze(1), labels.float())
                loss = (loss * valid_mask).mean()
            else:
                raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
