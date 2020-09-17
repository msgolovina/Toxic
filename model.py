import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, num_classes, config):
        super(BertClassifier, self).__init__(config)
        self.bert_model = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
    ):
        outputs = self.bert_model(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask,
        )
        cls_output = outputs[1]
        cls_output = self.classifier(cls_output)
        cls_output = torch.sigmoid(cls_output)
        return cls_output
