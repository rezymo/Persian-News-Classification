import torch.nn as nn
from transformers import BertForSequenceClassification


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-multilingual-cased"
        num_labels = 15
        self.encoder = BertForSequenceClassification.from_pretrained(options_name,
                                                                     num_labels=num_labels)

    def forward(self, text, label):

        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea