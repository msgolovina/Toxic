import torch
from torch import nn


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert_model = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

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
