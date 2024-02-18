from transformers import Trainer, TrainingArguments
from transformers import RobertaForTokenClassification, RobertaTokenizer, RobertaConfig, RobertaModel, BertForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from typing import Optional


# Create model
class RobertaForFrameIdentificationWithTargetToken(RobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, target_token_span: torch.LongTensor,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        """
        
        Args:
            target_token_span: (B, 2) tensor with target token positions
            **kwargs: kwargs for RobertaModel.forward()
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # Get target token representations
        target_token_representations = torch.stack([sequence_output[i, target_token_span[i, 0]:target_token_span[i, 1]+1, :].mean(dim=0) 
                                                    for i in range(sequence_output.shape[0])])

        logits = self.classifier(target_token_representations)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Create model
class BertForFrameIdentificationWithTargetToken(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, target_token_span: torch.LongTensor,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        """
        
        Args:
            target_token_span: (B, 2) tensor with target token positions
            **kwargs: kwargs for BertModel.forward()
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # Get target token representations
        target_token_representations = torch.stack([sequence_output[i, target_token_span[i, 0]:target_token_span[i, 1]+1, :].mean(dim=0) 
                                                    for i in range(sequence_output.shape[0])])

        logits = self.classifier(target_token_representations)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )