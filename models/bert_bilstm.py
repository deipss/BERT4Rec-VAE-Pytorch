from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTBILSTMModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.bert = BERT(args)
        nn.LSTM(input_size=self.bert.hidden,
                hidden_size=self.bert.hidden,
                num_layers=self.bert.bilstm_num,
                dropout=0.1,
                batch_first=True,
                bidirectional=True)
        self.out = nn.Linear(self.bert.hidden, args.num_items + 1)

    @classmethod
    def code(cls):
        return 'bert_bilstm'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)
